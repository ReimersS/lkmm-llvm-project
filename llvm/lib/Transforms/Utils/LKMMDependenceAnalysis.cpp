//===- LKMMDependenceAnalaysis.cpp - LKMM Deps Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements two passes to determine whether data, addr and ctrl
/// dependencies were preserved according to the Linux kernel memory model.
///
/// The first pass annotates relevant dependencies in unoptimized IR and the
/// second pass verifies that the dependenices still hold in optimized IR.
///
/// Linux kernel memory model:
/// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/memory-model/Documentation/explanation.txt
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LKMMDependenceAnalysis.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include <memory>
#include <unordered_set>

namespace llvm {
namespace {
static cl::opt<bool> InjectBugs(
    "lkmm-enable-tests",
    cl::desc("Enable the LKMM dependency checker tests. Requires the tests "
             "to be present in the source tree of the kernel being compiled"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> FullToPartialOpt(
    "enable-lkmm-addr-warnings",
    cl::desc("Enable warnings for LKMM addr dependencies based on full to "
             "partial addr dependency conversion"),
    cl::Hidden, cl::init(false));
} // namespace

// Avoid the std:: qualifier if possible
using std::list;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::unordered_map;
using std::unordered_set;

constexpr StringRef ADBStr = "LKMMDep: address dep begin";
constexpr StringRef ADEStr = "LKMMDep: address dep end";
constexpr StringRef PCsADBStr = "AddrDepBeginnings";
constexpr StringRef PCsADEStr = "AddrDepEndings";

// FIXME Is there a more elegant way of dealing with duplicate IDs
// (preferably getting eliminating the problem all together)?

// The IDReMap type alias represents the map of IDs to sets of alias IDs
// which verification contexts use for remapping duplicate IDs. Duplicate
// IDs appear when an annotated instruction is duplicated as part of
// optimizations.
using IDReMap = unordered_map<string, unordered_set<string>>;

// Represents a map of IDs to (potential) dependency halfs.
template <typename T> using DepHalfMap = unordered_map<string, T>;

/// Every dep chain link has a DCLevel. The level tracks whether the pointer
/// itself or the pointed-to value, the pointee, is part of the dependency
/// chain.
///
/// PTR   -> we're interested in the pointer itself.  PTE -> we're
/// interested in the pointed-to value.
///
/// BOTH  -> matches PTR __AND__ PTE.
///
/// NORET -> Dep chain doesn't get returned, but calling function should still
/// be made aware of its existence. The calling function then knows that the
/// beginning has been seen, but its dependency chain might have been broken.
///
/// EMPTY -> Empty.
enum class DCLevel { PTR, PTE, BOTH, NORET, EMPTY };

/// Represents a dependency chain link. A dep chain link consists of an IR
/// value and the corresponding dep chain level.
struct DCLink {
  DCLink(Value *Val, DCLevel Lvl) : Val(Val), Lvl(Lvl) {}
  Value *Val;
  DCLevel Lvl;

  bool operator==(const DCLink &Other) const {
    return Val == Other.Val && Lvl == Other.Lvl;
  }
};

/// Represents a dependency chain. A dep chain consists of a beginning, an
/// ending, and a unique chain of links between them.
struct DC {

  DC() = default;
  DC(DC &Other) : Chain(Other.Chain) {}

  void addLink(Value *Val, DCLevel Lvl) { Chain.push_back(DCLink(Val, Lvl)); }

  void addLink(DCLink Link) { Chain.push_back(Link); }

  // Links between (including) the beginning and the ending.
  // In reverse order; From the end to the beginning.
  list<DCLink> Chain;

  bool operator==(const DC &Other) const {
    return Chain == Other.Chain;
  }
};

/// Returns a string representation of an instruction's location in the form:
/// <function_name>::<line>:<column>.
///
/// \param I the instruction whose location string should be returned.
/// \param viaFile set to true if the filename should be used instead of the
///  function name
/// \param Entering set to true if the location for a call is being requested
/// which control is entering right now. In that case, line and column info
/// will remain the same, but the function name will be replaced with the
/// called function to make for better reading when outputting broken
/// dependencies.
///
/// \returns a string represenation of \p I's location.
string getInstLocString(Instruction *I, bool ViaFile = false) {
  const DebugLoc &InstDebugLoc = I->getDebugLoc();

  if (!InstDebugLoc)
    return "value with no source code location";

  auto LiAndCol = "::" + to_string(InstDebugLoc.getLine()) + ":" +
                  to_string(InstDebugLoc.getCol());

  if (ViaFile)
    return InstDebugLoc.get()->getFilename().str() + LiAndCol;

  return (I->getFunction()->getName().str()) + LiAndCol;
}

// Try to find dependencies bottom-up.

class BUCtx : public InstVisitor<BUCtx> {
public:
  enum CtxKind { CK_Annot };

  CtxKind getKind() const { return Kind; }

  BUCtx(Function *F, CtxKind CK)
      : F(F), Kind(CK){};

  void runSearch() {
    for (auto &BB : *F) {
      visitBasicBlock(BB);
    }
  }

  // Probably not needed.
  void visitBasicBlock(BasicBlock &BB);

  // End of a dep. chain. Beginning of search.
  void visitStore(StoreInst &SI);

  // Beginning of a dep. chain. End of search.
  void visitLoad(LoadInst &LI);
  
  // Helper function for visitLoad.
  void goThroughMem(LoadInst &LI);

  // Continue search through calls.
  //void visitCallInst(CallInst &CI);

  //void visitReturnInst(ReturnInst &ReturnI);

  // Continue search through other instructions.
  void visitUnaryOperator(UnaryOperator &UnOp) {};

  void visitBinaryOperator(BinaryOperator &BinOp) {};
 
  void visitExtractElementInst(ExtractElementInst &EEI) {};

  void visitInsertElementInst(InsertElementInst &IEI) {};

  void visitShuffleVectorInst(ShuffleVectorInst &SVI) {};

  void visitExtractValueInst(ExtractValueInst &EVI) {};

  void visitInsertValueInst(InsertValueInst &IVI) {};

  // This should never be reached with an incomplete chain?
  void visitAllocInst(AllocaInst &AI) {};

  // TODO:
  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &ACXI) {};

  // TODO:
  void visitAtomicRMWInst(AtomicRMWInst &ARMWI) {};

  void visitGetElementPtrInst(GetElementPtrInst &GEP);

  void visitPHINode(PHINode &PN) {};

  void visitTruncInst(TruncInst &TI) {};

  void visitZExtInst(ZExtInst &ZI) {};

  void visitSExtInst(SExtInst &SI) {};

  void visitPtrToIntInst(PtrToIntInst &PTI) {};

  void visitIntToPtrInst(IntToPtrInst &ITPI) {};

  void visitBitCastInst(BitCastInst &BCI) {};

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASCI) {};

  void visitSelectInst(SelectInst &SI) {};

protected:
  // The function the BFS is currently visiting.
  Function *F;

  // The BB the BFS is currently checking.
  BasicBlock *BB;

private:
  const CtxKind Kind;
};

class AnnotCtx : public BUCtx {
public:
  static bool classof(const BUCtx *C) { return C->getKind() == CK_Annot; }

  AnnotCtx(Function *F) : BUCtx(F, CK_Annot) {}

  /// Inserts the bugs in the testing functions. Will output to errs() if the
  /// desired annotation can't be found.
  ///
  /// \param F any testing function.
  /// \param IOpCode the type of Instruction whose dependency should be
  /// broken.
  ///  Can be Load or Store.
  /// \param AnnotationType the type of annotation to break, i.e. (addr +
  /// ctrl)
  ///  dep (beginning + ending).
  //void insertBug(Function *F, Instruction::MemoryOps IOpCode,
  //               string AnnotationType);

  void printIntactDeps(StringRef FName); // home

  void setNewDc(std::unique_ptr<DC> NewDC) {
    CurrDC = std::move(NewDC);
  }

  std::unique_ptr<DC> getDCPtr() {
    return std::move(CurrDC);
  }

  DC &getDc() {
    return *CurrDC;
  }

  void makeIntactDep() {
    //auto DCBegin = CurrDC->Chain.end()->Val;
    //auto DCEnd = CurrDC->Chain.begin()->Val;

    for (auto In = IntactDeps.begin(); In != IntactDeps.end(); In++){
      if (**In == *CurrDC) {
        return;
      }
    }
    std::string Annot;
    for (auto I = CurrDC->Chain.rbegin(); I != CurrDC->Chain.rend(); I++) {
      Annot += getInstLocString(cast<Instruction>(I->Val));
      if (I != CurrDC->Chain.rend()--)
        Annot += "-";
    }
    for (auto I = CurrDC->Chain.rbegin(); I != CurrDC->Chain.rend(); I++) {
      auto *Meta = MDNode::get(F->getContext(), MDString::get(F->getContext(), Annot));
      cast<Instruction>(I->Val)->setMetadata("annot", Meta);
    }
    IntactDeps.insert(std::move(CurrDC));
    errs() << "Intact dep: " << Annot << "\n";
  }

private:
  // Currently tracked DC.
  std::unique_ptr<DC> CurrDC;

  unordered_set<std::unique_ptr<DC>> IntactDeps;
};

// TODO: avoid back edges
void BUCtx::visitBasicBlock(BasicBlock &BB) {
  this->BB = &BB;

  for (auto &I : BB) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isVolatile())
        continue;

      auto *Ptr = SI->getPointerOperand();
    
      if (auto *Up = dyn_cast<Instruction>(Ptr)) {

        auto *Ann = (AnnotCtx *)this;
        auto End = std::make_unique<DC>();
        End->addLink(SI, DCLevel::PTR);
        Ann->setNewDc(std::move(End));
        visit(Up);
      }
    }
  }
}

void BUCtx::visitStore(StoreInst &SI) {
  // We might end up here because we stored the linking val to mem
  if (this->Kind == CK_Annot) {
    auto *Ann = (AnnotCtx *)this;
    auto *Val = dyn_cast<Value>(&(SI));

    if (Ann->getDc().Chain.rbegin()->Lvl == DCLevel::PTE) {
      Ann->getDc().addLink(Val, DCLevel::PTR);
      auto *Val = SI.getValueOperand();
      if (auto *Up = dyn_cast<Instruction>(Val)) {
        visit(Up);
      }
    }
  }
}

void BUCtx::visitLoad(LoadInst &LI) {
  if (!LI.isVolatile()) {
    goThroughMem(LI);
    return;
  }

  if (this->Kind == CK_Annot) {
    auto *Ann = (AnnotCtx *)this;
    
    auto *Val = dyn_cast<Value>(&(LI));

    Ann->getDc().addLink(Val, DCLevel::PTR);
    Ann->makeIntactDep();
  }
}

void BUCtx::goThroughMem(LoadInst &LI) {

  auto *Ann = (AnnotCtx *)this;
  auto *Val = dyn_cast<Value>(&(LI));

  assert(Ann->getDc().Chain.rbegin()->Lvl == DCLevel::PTR &&
         "Expected a pointer to be the last link in the chain for Load");
  Ann->getDc().addLink(Val, DCLevel::PTE);

  // Find previous stores that write the same location and continue there.
  for (auto *U : LI.getPointerOperand()->users()) {
    // TODO: move Curr out of loop
    auto Curr = Ann->getDCPtr();
    auto Cpy = std::make_unique<DC>(*Curr);

    Ann->setNewDc(std::move(Cpy));
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == LI.getPointerOperand()) {
        visit(SI);
      }
    }
    Ann->setNewDc(std::move(Curr));
  }
}

void BUCtx::visitGetElementPtrInst(GetElementPtrInst &GEP) {

  // GEP is a glorified add
  auto *Ann = (AnnotCtx *)this;
  auto *Val = dyn_cast<Value>(&(GEP));

  assert(Ann->getDc().Chain.rbegin()->Lvl == DCLevel::PTR &&
         "Expected a pointer to be the last link in the chain for GEP");

  Ann->getDc().addLink(Val, DCLevel::PTR);

  auto Curr = Ann->getDCPtr();

  // Track all indexes
  for (auto &Idx : GEP.indices()) {
    if (auto *Up = dyn_cast<Instruction>(Idx)) {
      auto Cpy = std::make_unique<DC>(*Curr);
      Ann->setNewDc(std::move(Cpy));
      visit(Up);
    }
  }

  // Track the pointer
  if (auto *Up = dyn_cast<Instruction>(GEP.getPointerOperand())) {
    Ann->setNewDc(std::move(Curr));
    visit(Up);
  }
}

class LKMMAnnotator {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class LKMMVerifier {
public:
  //LKMMVerifier();

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  // Contains all unverified address dependency beginning annotations.
  // shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;

  // Contains all unverified address dependency ending annotations.
  // shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;

  //shared_ptr<IDReMap> RemappedIDs;

  //shared_ptr<unordered_set<string>> VerifiedIDs;

  //unordered_set<string> PrintedBrokenIDs;

  //unordered_set<Module *> PrintedModules;

  /// Maps the reduced IDs of the same beginning / ending to the shortest
  /// VerAddDepBeg with that ending plus the length of its ID.  An ID is
  /// reduced if it excludes the path from the beginning to the end and only
  /// contains the beginning location and the ending location.
  // StringMap<pair<VerAddrDepBeg *, unsigned>> MinLengthPerBegEndPair;

  /// Prints broken dependencies.
  //void printBrokenDeps();

  // void printBrokenDep(VerDepHalf &Beg, VerDepHalf &End, const string &ID);

};

PreservedAnalyses LKMMAnnotator::run(Module &M, ModuleAnalysisManager &AM) {
  // FIXME: Come up with a way of making bug insertion upstream compatible
  bool InsertedBugs = false;

  for (auto &F : M) {
    if (F.empty())
      continue;

    AnnotCtx AC(&F);

    // Annotate dependencies.
    AC.runSearch();

    //AC.printIntactDeps();

    if (InjectBugs) {
      if (!F.hasName())
        continue;

      auto FName = F.getName();

      // Insert bugs if the BFS just annotated a testing function.
      if (FName.contains("proj_bdo_rr_addr_dep_begin") ||
          FName.contains("proj_bdo_rw_addr_dep_begin") ||
          FName.contains("proj_bdo_ctrl_dep_begin")) {
        //AC.insertBug(&F, Instruction::Load, "dep begin");
        InsertedBugs = true;
      }

      // Break read -> read addr dep endings.
      else if (FName.contains("proj_bdo_rr_addr_dep_end")) {
        //AC.insertBug(&F, Instruction::Load, "dep end");
        InsertedBugs = true;
      }

      // Break read -> write addr dep and ctrl dep endings.
      else if (FName.contains("proj_bdo_rw_addr_dep_end") ||
               FName.contains("proj_bdo_ctrl_dep_end")) {
        //AC.insertBug(&F, Instruction::Store, "dep end");
        InsertedBugs = true;
      }
    }
  }

  return InsertedBugs ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses LKMMVerifier::run(Module &M, ModuleAnalysisManager &AM) {
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses LKMMAnnotateDepsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto A = LKMMAnnotator();

  return A.run(M, AM);
}

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses LKMMVerifyDepsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto V = LKMMVerifier();

  return V.run(M, AM);
}

} // namespace llvm
