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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
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

// This list is complete and will never change
#define FOR_EACH_DEP(DO) \
  DO(Intact) \
  DO(Rising) \
  DO(MayDangle) \
  DO(Dangling) \
  DO(RisingDangling) \
  DO(MayDangleDangling) \
  DO(MayRise) \
  DO(MayRiseRising) \
  DO(MayRiseMayDangle)


// Avoid the std:: qualifier if possible
using std::string;
using std::unordered_set;

namespace llvm {

std::string getInstLocString(Instruction *I, bool ViaFile) {
  const DebugLoc &InstDebugLoc = I->getDebugLoc();

  if (!InstDebugLoc)
    return "value with no source code location";

  auto LiAndCol = "::" + std::to_string(InstDebugLoc.getLine()) + ":" +
                  std::to_string(InstDebugLoc.getCol());

  if (ViaFile)
    return InstDebugLoc.get()->getFilename().str() + LiAndCol;

  return (I->getFunction()->getName().str()) + LiAndCol;
}

std::string getInstLocString(const StringRef &F ,const DebugLoc &InstDebugLoc, bool ViaFile) {
  if (!InstDebugLoc)
    return "value with no source code location";

  auto LiAndCol = "::" + std::to_string(InstDebugLoc.getLine()) + ":" +
                  std::to_string(InstDebugLoc.getCol());

  if (ViaFile)
    return InstDebugLoc.get()->getFilename().str() + LiAndCol;

  return (F.str()) + LiAndCol;
}

/// Represents a dependency chain link on LLVM IR level. A dep chain link consists of an IR
/// instruction and the corresponding dep chain level.
///
/// This is private to the LKMMAnnotateDepsPass, as values may have no meaning
/// after other optimisation passes
class LKMMAnnotateDepsPass::DCLink : public DCLinkBase {
public:
  DCLink(Instruction *Val, const DCLevel Lvl) : DCLinkBase(&Val->getDebugLoc(), Lvl), Val(Val) {}
  ~DCLink() = default;

  DCLink(const DCLink &Other) : DCLinkBase(Other.Loc, Other.Lvl), Val(Other.Val) {}

  Instruction *Val;

  bool isCall() const { return CallInst::classof(Val); }
  bool isRet() const { return ReturnInst::classof(Val); }

  bool operator==(const DCLinkBase &Other) const override {
    const auto &O = static_cast<const DCLink &>(Other);
    return Val == O.Val && Lvl == O.Lvl && Depth == O.Depth;
  }
};

/// Represents a dependency chain link on source level. A dep chain link consists of a
/// source code location, the corresponding dep chain level, and the function call depth.
class LKMMAnnotateDeps::DCLink : public DCLinkBase {

public:
  //DCLink(DebugLoc &Loc, DCLevel Lvl, DCLinkType Type = DCLinkType::VALUE) : DCLinkBase(Loc, Lvl), Depth(0), Type(Type) {}
  ~DCLink() = default;

  // Convenience copy constructor
  DCLink(const LKMMAnnotateDepsPass::DCLink &Other) : DCLinkBase(Other.Loc, Other.Lvl, Other.getDepth()), F(Other.Val->getFunction()), Type(DCLinkType::VALUE) {
    if (Other.isCall()) Type = DCLinkType::CALL;
    if (Other.isRet()) Type = DCLinkType::RETURN;
  }

  bool isCall() const { return Type == DCLinkType::CALL; }
  bool isRet() const { return Type == DCLinkType::RETURN; }

  bool operator==(const DCLinkBase &Other) const override {
    const auto &O = static_cast<const DCLink &>(Other);
    return Loc->getLine() == O.Loc->getLine() && Loc->getCol() == O.Loc->getCol() && Depth == O.Depth;
  }

  // Ok to keep pointers to functions.
  // a) Only needed to annotate the IR chains (so definetely valid)
  // b) Optimizations are unlikely to delete entire functions
  const Function *F;
private:
  DCLinkType Type;
};

/// Represents a dependency chain (segment). A dep chain consists of a beginning, an
/// ending, and a unique chain of links between them.
///
/// We use the names "dependency chain" and "chain segment" interchangeably.
template <typename Context>
struct DC {
  DC() {};

  //DC(const DC<Context> &Other) : Chain(Other.Chain), ArgB(Other.ArgB), ArgE(Other.ArgE) {};

  template <typename T>
  DC<Context>(const DC<T> &Other) = delete;

  DC(DC &A, DC &B, int Delta) = delete;

  void addLink(const typename Context::DCLink &Link, std::optional<int> Arg = std::nullopt) {

    // In source level chains, we only add Links with a location.
    // This can happen when declaring local variables.
    if constexpr (std::is_same_v<Context, LKMMAnnotateDeps>) {
      if (!Link.Loc->get())
        return;
    }

    if (Chain.empty()) {
      ArgE = Arg;
      Chain.push_back(Link);
      return;
    }

    if (Link == Chain.back())
      return;

    ArgB = Arg;
    Chain.push_back(Link);
  }

  void addLink(Instruction *Val, DCLevel Lvl, std::optional<int> Arg = std::nullopt) = delete;

  // Links between (including) the beginning and the ending.
  // In reverse order; from the end to the beginning.
  std::vector<typename Context::DCLink> Chain;

  // Both segments begin in a call inst;
  // may dangle: we tracked up to the value of this call in F
  // rises: we tracked up to the begining of F and stored one specific call site (likely not in F)
  bool mayDangle() {
    return Chain.back().isCall() && !ArgB;
  }
  bool rises() {
    return Chain.back().isCall() && ArgB;
  }

  // Segment ends in a call
  // ArgE must have a value
  bool mayRise() {
    return Chain.front().isCall();
  }

  // Segment ends in a return
  bool dangles() {
    return Chain.front().isRet();
  }

  // Chain does not begin or end in the function.
  // The escaping arguments must be annotated.
  std::optional<int> ArgB;
  std::optional<int> ArgE;

  bool operator==(const DC &Other) const {
    return Chain == Other.Chain && ArgB == Other.ArgB && ArgE == Other.ArgE;
  }
};

/// Convenience specialization.
/// Adds a value to the IR level dependency chain.
template<>
void DC<LKMMAnnotateDepsPass>::addLink(Instruction *Val, DCLevel Lvl, std::optional<int> Arg) {
  LKMMAnnotateDepsPass::DCLink Link(Val, Lvl);
  addLink(Link, Arg);
}

// No need to check for compatibility, the segment specialization
// should ensure this.
/// Concatenates two source level dependency chains.
/// Merging must be done before annotation, otherwise we lose access to the instructions.
template<>
DC<LKMMAnnotateDepsPass>::DC(DC &Beg, DC &End, int Delta) {

  // keep in mind that the chains are in reverse order
  auto It = Chain.insert(Chain.begin(), End.Chain.begin(), End.Chain.end());
  // Chain: [End.E, ...., End.B]
  if (Delta < 0) {
    for (auto &I = It; I != Chain.end(); I++) {
      I->addDepth(-Delta);
    }
  }
  It = Chain.insert(Chain.end(), Beg.Chain.begin(), Beg.Chain.end());
  // Chain: [End.E, ...., End.B, Beg.E, ...., Beg.B]
  if (Delta > 0) {
    for (auto &I = It; I != Chain.end(); I++) {
      I->addDepth(Delta);
    }
  }
  ArgB = Beg.ArgB;
  ArgE = End.ArgE;
}

/// Convenience specialization.
/// One-way copy constructor from an IR level chain, to a source level chain.
template<>
template<>
DC<LKMMAnnotateDeps>::DC(const DC<LKMMAnnotateDepsPass> &Other) {
  for (const auto &Link : Other.Chain) {
    auto NewLink = LKMMAnnotateDeps::DCLink(Link);
    addLink(NewLink);
  }
  ArgB = Other.ArgB;
  ArgE = Other.ArgE;
}

// All 9 combinations of chain segments.
//
// Pairs (Begin, End) with values: 0=internal, -1=caller, +1=callee.
//  0  0 -> intact                  (trivial)
// -1  0 -> rising                  (arg -> X_ONCE)
// +1  0 -> may dangle              (call -> X_ONCE)
//
//  0 -1 -> dangling                (X_ONCE -> return)
// -1 -1 -> rising & dangling       (arg -> return)
// +1 -1 -> may dangle & dangling   (call -> return)
//
//  0 +1 -> may rise                (X_ONCE -> call)
// -1 +1 -> may rise & rising       (arg -> call)
// +1 +1 -> may rise & may dangle   (call -> call)
//
// WHY:
// There are VERY limited options for combining segments, but potentially infinitely long chains.
// Each intact chain begins and ends with 0 (potentially different pairs though)
// Each +1 end must continue in a -1 begin.
// Each -1 end must continue in a +1 begin.
// A chains absolute, summed up delta at pair N must be smaller or equal to 2N
// (the fastest we can approach 0 is in steps of 2)
//
// FIXME: There is probably integer polynomial wizardry going on that could prove complexity and optimality
//
// FIXME: remove getters that make no sense
template<> const std::string SegmentID<0, 0>::Type = "Intact";
template<> const std::string SegmentID<-1, 0>::Type = "Rising";
template<> const std::string SegmentID<1, 0>::Type = "May Dangle";
template<> const std::string SegmentID<0, -1>::Type = "Dangling";
template<> const std::string SegmentID<-1, -1>::Type = "Rising & Dangling";
template<> const std::string SegmentID<1, -1>::Type = "May Dangle & Dangling";
template<> const std::string SegmentID<0, 1>::Type = "May Rise";
template<> const std::string SegmentID<-1, 1>::Type = "May Rise & Rising";
template<> const std::string SegmentID<1, 1>::Type = "May Rise & May Dangle";

// Try to find dependencies bottom-up.

class BUCtx : public InstVisitor<BUCtx> {
public:
  enum CtxKind { CK_Annot };

  CtxKind getKind() const { return Kind; }

  BUCtx(CtxKind CK)
      : Kind(CK){};

  void runSearch() {
    for (auto &BB : *F) {
      visitBasicBlock(BB);
    }
  }

  // Generic forwarder for all values.
  void visit(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      InstVisitor::visit(I);
    }
    if (auto *A = dyn_cast<Argument>(V)) {
      visitArgument(A);
    }
  }

  // Helper for segments that begin with the current function.
  void visitArgument(Argument *A);

  // Probably not needed.
  void visitBasicBlock(BasicBlock &BB);

  // Continues search through mem.
  // Cannot be the end of chain, this is handled in visitBB (Pass 1).
  void visitStore(StoreInst &SI);

  // Potential beginning of a dep chain.
  // May end current search, always continues through mem.
  void visitLoad(LoadInst &LI);
  
  // Helper function for visitLoad.
  void goThroughMem(LoadInst &LI);

  // Beginning of a "may dangle" segment. End of search.
  // Cannot be the end of a "may rise" segment, this is handled in visitBB (Pass 3).
  void visitCallInst(CallInst &CI);

  // Not needed, we explicitly start from the returned values in visitBB (Pass 2).
  //void visitReturnInst(ReturnInst &ReturnI);

  // Continue search through other instructions.
  void visitUnaryOperator(UnaryOperator &UnOp) {};

  void visitBinaryOperator(BinaryOperator &BinOp) {};
 
  void visitExtractElementInst(ExtractElementInst &EEI) {};

  void visitInsertElementInst(InsertElementInst &IEI) {};

  void visitShuffleVectorInst(ShuffleVectorInst &SVI) {};

  void visitExtractValueInst(ExtractValueInst &EVI) {};

  void visitInsertValueInst(InsertValueInst &IVI) {};

  // TODO: This should end the search. Pointers to local memory cannot begin chains.
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

class LKMMAnnotateDepsPass::AnnotCtx : public BUCtx {
public:
  static bool classof(const BUCtx *C) { return C->getKind() == CK_Annot; }

  AnnotCtx() : BUCtx(CK_Annot), Result({}), CurrPass(Pass::Known_End) {};

  void printIntactDeps(StringRef FName);

  void setNewDc(std::unique_ptr<DC> NewDC) {
    CurrDC = std::move(NewDC);
  }

  std::unique_ptr<DC> getDCPtr() {
    return std::move(CurrDC);
  }

  DC &getDc() {
    return *CurrDC;
  }

  SmallVector<llvm::DC<LKMMAnnotateDeps>> getResult() { return Result; }

  // Only runs once. Annotates ALL segments ending in volatile loads and stores.
  void passOne(Function *NewF, IntactDeps_t *I, RisingDeps_t *R, MayDangleDeps_t *MD) {
    F = NewF;
    this->I = I;
    this->R = R;
    this->MD = MD;

    runSearch();
  }

  // TODO: Put 2 & 3 (& 4?!) in a fixed point loop

  // Runs on all functions with RetAttr "returns_X_dep".
  // May add more segments with the any attr.
  void passTwo(Function *NewF, DanglingDeps_t *D, RisingDanglingDeps_t *RD, MayDangleDanglingDeps_t *MDD) {

    F = NewF;
    this->D = D;
    this->RD = RD;
    this->MDD = MDD;

    CurrPass = Pass::Known_Ret;
    runSearch();
  }
  
  // Runs on all functions with FnAttr "takes_X_dep".
  // May add more segments with the any attr.
  void passThree(Function *NewF, MayRiseDeps_t *MR, MayRiseRisingDeps_t *MRR, MayRiseMayDangleDeps_t *MRMD) {

    F = NewF;
    this->MR = MR;
    this->MRR = MRR;
    this->MRMD = MRMD;

    CurrPass = Pass::Known_Call;
    runSearch();
  }

  // Merges all segments to full dependency chains of length Depth.
  void merge(const size_t &Depth) {
    for (size_t D = 1; D <= Depth; D++)
      buildTransitiveClosure(D);
  }

  template<int B, int E>
  void makeIntactDep() {

    DepMap<B, E> *TypedMap;

    auto Seg = SegmentID<B, E>(getDc());

    if constexpr (B == 0 && E == 0) {
      TypedMap = I;
    } else if constexpr (B == -1 && E == 0) {
      TypedMap = R;
    } else if constexpr (B == 1 && E == 0) {
      TypedMap = MD;
    } else if constexpr (B == 0 && E == -1) {
      TypedMap = D;
    } else if constexpr (B == -1 && E == -1) {
      TypedMap = RD;
    } else if constexpr (B == 1 && E == -1) {
      TypedMap = MDD;
    } else if constexpr (B == 0 && E == 1) {
      TypedMap = MR;
    } else if constexpr (B == -1 && E == 1) {
      TypedMap = MRR;
    } else if constexpr (B == 1 && E == 1) {
      TypedMap = MRMD;
    }

    if (TypedMap->find(Seg) == TypedMap->end()) {
      std::unordered_set<std::unique_ptr<DC>> DCSet;
      TypedMap->insert({Seg, std::move(DCSet)});
    }

    if constexpr (B == -1) {
      // Attention! The beginning call instruction _calls_ F. It probably is not in F.
      auto *CallingInstr = cast<CallInst>(CurrDC->Chain.back().Val);
      auto *Caller = CallingInstr->getFunction();

      Caller->addFnAttr(Attribute::get(Caller->getContext(), "calls_addr_dep"));
      F->addFnAttr(Attribute::get(F->getContext(), "takes_addr_dep"));
      F->addParamAttr(CurrDC->ArgB.value(), Attribute::get(F->getContext(), "is_addr_dep"));
    }
    
    if constexpr (B == 1) {
      // Attention! The beginning return returns to F and
      // definitely returns to it (else everything we just traversed is unreachable).

      auto *Callee = cast<ReturnInst>(CurrDC->Chain.back().Val)->getFunction();
      Callee->addRetAttr(Attribute::get(F->getContext(), "returns_addr_dep"));
      // TODO: Warn about external function/intrinsic
    }

    // Endings are already known once we start the appropriate pass.
    // Only do sanity checks here.
    if constexpr (E == -1) {
      
      // TODO: rebase to newest llvm
      assert(F->getAttributes().getRetAttrs().hasAttribute("returns_addr_dep") && "Function should not have been passed in Pass 2");
    }

    if constexpr (E == 1) {
      
      auto *Callee = cast_if_present<CallInst>(CurrDC->Chain.front().Val);
      if (Callee) {
        assert(Callee->getCalledFunction()->getAttributes().getFnAttrs().hasAttribute("takes_addr_dep") && "Function should not have been passed in Pass 3");
        assert(Callee->getCalledFunction()->getAttributes().getParamAttrs(CurrDC->ArgE.value()).hasAttribute("is_addr_dep") && "Argument should not have been passed in Pass 3");
      }
    }

    TypedMap->at(Seg).insert(std::move(CurrDC));
  }

  enum Pass { Known_End, Known_Ret, Known_Call, Match };

  Pass currPass() const { return CurrPass; }

private:
  // Currently tracked DC.
  std::unique_ptr<DC> CurrDC;

  SmallVector<llvm::DC<LKMMAnnotateDeps>> Result;

  // Current annotation pass
  Pass CurrPass;

  // Pass 1, can never be extended
  IntactDeps_t *I;
  RisingDeps_t *R;
  MayDangleDeps_t *MD;

  // Pass 2 & 3, can be extended by any segment with an even delta (0 or 2)
  DanglingDeps_t *D;
  RisingDanglingDeps_t *RD;
  MayDangleDanglingDeps_t *MDD;
  MayRiseDeps_t *MR;
  MayRiseRisingDeps_t *MRR;
  MayRiseMayDangleDeps_t *MRMD;
  
  void buildTransitiveClosure(const size_t Depth);
  void annotateChain(DC &C);
  template<int B, int M, int E>
  DepMap<B,E> match(DepMap<B, M> *Beg, DepMap<-M, E> *End);
};

/// Converts an LLVM IR level chain to a source level chain, and
/// annotates the chain in the IR.
void LKMMAnnotateDepsPass::AnnotCtx::annotateChain(DC &C) {
    
    std::string Annot;
    std::string Pretty;

    llvm::DC<LKMMAnnotateDeps> Ret = C;

    for (auto I = Ret.Chain.crbegin(); I != Ret.Chain.crend(); I++) {
      Annot += getInstLocString(I->F->getName(), *I->Loc);
      Pretty += std::string(I->getDepth(), '\t') + getInstLocString(I->F->getName(), *I->Loc);
      if (I != std::prev(Ret.Chain.crend())) {
        Annot += "--";
        Pretty += "\n";
      }
    }

    for (auto I = C.Chain.crbegin(); I != C.Chain.crend(); I++) {
      MDNode *Meta = MDNode::get(I->Val->getContext(), MDString::get(I->Val->getContext(), Annot));

      if (auto *Existing = I->Val->getMetadata("addr_dep"))
        Meta = llvm::MDNode::concatenate(Existing, Meta);
      I->Val->setMetadata("addr_dep", Meta);
    }

    errs() << "[Annotation] Complete Chain:\n" << Pretty << "\n\n";
    
    Result.push_back(Ret);
}

template<int B, int M, int E>
LKMMAnnotateDepsPass::DepMap<B,E> LKMMAnnotateDepsPass::AnnotCtx::match(DepMap<B, M> *Beg, DepMap<-M, E> *End) {
  if (!Beg || !End)
    return DepMap<B,E>();

  DepMap<B,E> Ret;
  for (auto &[EndSeg, EndPtrs] : *End) {
    for (auto &[BegSeg, BegPtrs] : *Beg) {

      if (BegSeg.isCompatible(EndSeg)) {
        for (auto &EndDC : EndPtrs) {
          for (auto &BegDC : BegPtrs) {
            auto Dc = std::make_unique<DC>(*BegDC, *EndDC, BegSeg.getE());
            Ret[SegmentID<B,E>(*Dc)].insert(std::move(Dc));
          }
        }
      }
    }
  }
  return Ret;
}

void LKMMAnnotateDepsPass::AnnotCtx::buildTransitiveClosure(const size_t Depth) {

  if (Depth == 1) {
    // Depth 1: Trivial
    errs() << std::remove_reference_t<decltype(*I)>::key_type::Type << ":\n";
    for (auto &Seg : *I) {
      for (auto It = Seg.second.begin(); It != Seg.second.end(); It++) {
        // We do not need the IR level chains after this point.
        annotateChain(**It);
      }
    }
    return;
  }

  if (Depth == 2) {
    // Depth 2: Still trivial, concatenate all matching R/MR and MD/D pairs
    errs() << std::remove_reference_t<decltype(*MR)>::key_type::Type << ", ";
    errs() << std::remove_reference_t<decltype(*R)>::key_type::Type << ":\n";
    auto DCs = match(MR, R);
    for (auto &[_, DCPtrs] : DCs) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      annotateChain(**It);
    }

    errs() << std::remove_reference_t<decltype(*D)>::key_type::Type << ", ";
    errs() << std::remove_reference_t<decltype(*MD)>::key_type::Type << ":\n";
    DCs = match(D, MD);
    for (auto &[_, DCPtrs] : DCs) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      annotateChain(**It);
    }

    return;
  }

  // We start with <X, 0> Chains of length 1
  // We match all <Y, -X> with Y!=0 to <Y, 0> Chains of length 2
  // Repeat until Depth-1
  // Add <0, -Y> to complete the chains
  auto PrevPos = std::make_unique<DepMap<1, 0>>();
  auto PrevNeg = std::make_unique<DepMap<-1, 0>>();
  auto CurPos = std::make_unique<DepMap<1, 0>>();
  auto CurNeg = std::make_unique<DepMap<-1, 0>>();
  size_t Len = 1;

  for (auto &[Seg, DCPtrs] : match(MRR, R)) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    (*PrevNeg)[Seg].insert(std::make_unique<DC>(**It));
  }
  for (auto &[Seg, DCPtrs] : match(MRMD, R)) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    (*PrevPos)[Seg].insert(std::make_unique<DC>(**It));
  }
  for (auto &[Seg, DCPtrs] : match(MDD, MD)) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    (*PrevPos)[Seg].insert(std::make_unique<DC>(**It));
  }
  for (auto &[Seg, DCPtrs] : match(RD, MD)) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    (*PrevNeg)[Seg].insert(std::make_unique<DC>(**It));
  }

  // TODO: check for delta
  while (Len <= Depth-1) {
    for (auto &[Seg, DCPtrs] : match(MRR, PrevNeg.get())) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      (*PrevNeg)[Seg].insert(std::make_unique<DC>(**It));
    }
    for (auto &[Seg, DCPtrs] : match(MRMD, PrevNeg.get())) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      (*PrevPos)[Seg].insert(std::make_unique<DC>(**It));
    }
    for (auto &[Seg, DCPtrs] : match(MDD, PrevPos.get())) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      (*PrevPos)[Seg].insert(std::make_unique<DC>(**It));
    }
    for (auto &[Seg, DCPtrs] : match(RD, PrevPos.get())) {
      for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
      (*PrevNeg)[Seg].insert(std::make_unique<DC>(**It));
    }

    PrevNeg = std::move(CurNeg);
    PrevPos = std::move(CurPos);
    CurNeg = std::make_unique<DepMap<-1, 0>>();
    CurPos = std::make_unique<DepMap<1, 0>>();
    Len++;
  }

  for (auto &[_, DCPtrs] : *PrevNeg) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    annotateChain(**It);
  }
  for (auto &[_, DCPtrs] : *PrevNeg) {
    for (auto It = DCPtrs.begin(); It != DCPtrs.end(); It++)
    annotateChain(**It);
  }
}

// TODO: avoid back edges
void BUCtx::visitBasicBlock(BasicBlock &BB) {
  this->BB = &BB;
  auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

  if (Ann->currPass() == LKMMAnnotateDepsPass::AnnotCtx::Pass::Known_End) {
    for (auto &I : BB) {
      Value *Ptr = nullptr;

      // Address dependencies end in a volatile load/store
      // with the ptr operand being the end of the chain.
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isVolatile())
          continue;

        Ptr = SI->getPointerOperand();
      }
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isVolatile())
          continue;

        Ptr = LI->getPointerOperand();
      }
      
      if (Ptr) {

          auto End = std::make_unique<DC<LKMMAnnotateDepsPass>>();
          End->addLink(&I, DCLevel::PTR);
          Ann->setNewDc(std::move(End));
          visit(Ptr);
      }
    }
      return;
  } // !Known_End

  if (Ann->currPass() == LKMMAnnotateDepsPass::AnnotCtx::Pass::Known_Ret) {
    // We also need to track any potential chains from return values.
    // FIXME: Aggregate returns should have an annotation per element.
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      for (auto &Op : RI->operands()) {
        auto End = std::make_unique<DC<LKMMAnnotateDepsPass>>();
        if (Op->getType()->isPointerTy())
          End->addLink(RI, DCLevel::PTR);
        else
          End->addLink(RI, DCLevel::PTE);
        Ann->setNewDc(std::move(End));
        visit(Op);
      } 
    }
    return;
  } // !Known_Ret

  if (Ann->currPass() == LKMMAnnotateDepsPass::AnnotCtx::Pass::Known_Call) {
    for (auto &I : BB) {
    // We also need to track any potential chains from call isntructions.
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        for (auto &Arg : CI->args()) {
          auto ArgNo = CI->getArgOperandNo(&Arg);
          auto End = std::make_unique<DC<LKMMAnnotateDepsPass>>();
          // FIXME: this might be wrong
          if (Arg->getType()->isPointerTy())
            End->addLink(&I, DCLevel::PTR, ArgNo);
          else
            End->addLink(&I, DCLevel::PTE, ArgNo);
          Ann->setNewDc(std::move(End));
          visit(Arg);
        }
      }
    }
    return;
  } // !Known_Call
  
  llvm_unreachable("Unknown Annotation Pass");
}

void BUCtx::visitArgument(Argument *A) {
  auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

  // We found a rising segment!
  // Add a new segment for all call sites of F (likely outside of F)

  for (auto *CallingInstr : F->users()) {
    if (auto *CI = dyn_cast<CallInst>(CallingInstr)) {

      auto Curr = Ann->getDCPtr();
      auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);
      
      Ann->setNewDc(std::move(Cpy));
      Ann->getDc().addLink(CI, Curr->Chain.back().Lvl, A->getArgNo());

      if (auto *_ = dyn_cast<ReturnInst>(Curr->Chain.front().Val))
        Ann->makeIntactDep<-1, -1>();
      else if (auto *_ = dyn_cast<CallInst>(Curr->Chain.front().Val))
        Ann->makeIntactDep<-1, 1>();
      else
        Ann->makeIntactDep<-1, 0>();
      
      Ann->setNewDc(std::move(Curr));
    }
  }
}

void BUCtx::visitStore(StoreInst &SI) {
  // We might end up here because we stored the linking val to mem
  if (this->Kind == CK_Annot) {
    auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

    if (Ann->getDc().Chain.crbegin()->Lvl == DCLevel::PTE) {
      Ann->getDc().addLink(&SI, DCLevel::PTR);
      auto *Val = SI.getValueOperand();
      visit(Val);
    }
  }
}

void BUCtx::visitLoad(LoadInst &LI) {
  if (this->Kind == CK_Annot) {

    if (LI.isVolatile()) {
      // We found an internal beginning!
      auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;
      
      // Save a copy without the load
      auto Curr = Ann->getDCPtr();
      auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);
      
      Ann->setNewDc(std::move(Cpy));
      Ann->getDc().addLink(&LI, DCLevel::PTR);
      if (auto *_ = dyn_cast<ReturnInst>(Curr->Chain.front().Val))
        Ann->makeIntactDep<0, -1>();
      else if (auto *_ = dyn_cast<CallInst>(Curr->Chain.front().Val))
        Ann->makeIntactDep<0, 1>();
      else
        Ann->makeIntactDep<0, 0>();

      Ann->setNewDc(std::move(Curr));
    }
    goThroughMem(LI);
  }
}

void BUCtx::goThroughMem(LoadInst &LI) {

  auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

  // TODO: Are double load/stores ok?
  // Sounds like aliasing
  if (Ann->getDc().Chain.crbegin()->Lvl == DCLevel::PTE) return;

  assert(Ann->getDc().Chain.crbegin()->Lvl == DCLevel::PTR &&
         "Expected a pointer to be the last link in the chain for Load");
  Ann->getDc().addLink(&LI, DCLevel::PTE);

  // Find previous stores that write the same location and continue there.
  // Anything else is potential alias territory (conservatively speaking)
  for (auto *U : LI.getPointerOperand()->users()) {
    // TODO: move Curr out of loop
    auto Curr = Ann->getDCPtr();
    auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);

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
  auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

  assert(Ann->getDc().Chain.crbegin()->Lvl == DCLevel::PTR &&
         "Expected a pointer to be the last link in the chain for GEP");

  Ann->getDc().addLink(&GEP, DCLevel::PTR);

  auto Curr = Ann->getDCPtr();

  // Track all indexes
  for (auto &Idx : GEP.indices()) {
    auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);
    Ann->setNewDc(std::move(Cpy));
    visit(Idx);
  }

  // Track the pointer
  Ann->setNewDc(std::move(Curr));
  visit(GEP.getPointerOperand());
}

void BUCtx::visitCallInst(CallInst &CI) {

  auto *Ann = (LKMMAnnotateDepsPass::AnnotCtx *)this;

  // We found segments that may dangle!
  // Add a new segment for all returns in the callee.

  Ann->getDc().addLink(&CI, Ann->getDc().Chain.back().Lvl);

  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return;

  for (auto &BB : *Callee) {
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      auto Curr = Ann->getDCPtr();
      auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);
      
      Ann->setNewDc(std::move(Cpy));
      Ann->getDc().addLink(RI, Curr->Chain.back().Lvl);

      auto *End = Ann->getDc().Chain.front().Val;
      if (auto *_ = dyn_cast<ReturnInst>(End))
        Ann->makeIntactDep<1, -1>();
      else if (auto *_ = dyn_cast<CallInst>(End))
        Ann->makeIntactDep<1, 1>();
      else
        Ann->makeIntactDep<1, 0>();

      Ann->setNewDc(std::move(Curr));
    }
  }
}

class LKMMAnnotateDepsPass::LKMMAnnotator {
public:
  LKMMAnnotator() : IntactDeps(std::make_unique<IntactDeps_t>()),
                    RisingDeps(std::make_unique<RisingDeps_t>()),
                    MayDangleDeps(std::make_unique<MayDangleDeps_t>()),
                    DanglingDeps(std::make_unique<DanglingDeps_t>()),
                    RisingDanglingDeps(std::make_unique<RisingDanglingDeps_t>()),
                    MayDangleDanglingDeps(std::make_unique<MayDangleDanglingDeps_t>()),
                    MayRiseDeps(std::make_unique<MayRiseDeps_t>()),
                    MayRiseRisingDeps(std::make_unique<MayRiseRisingDeps_t>()),
                    MayRiseMayDangleDeps(std::make_unique<MayRiseMayDangleDeps_t>()),
                    Stats({}) {};

  SmallVector<llvm::DC<LKMMAnnotateDeps>> run(Module &M, ModuleAnalysisManager &AM);

private:

  std::unique_ptr<IntactDeps_t> IntactDeps;
  std::unique_ptr<RisingDeps_t> RisingDeps;
  std::unique_ptr<MayDangleDeps_t> MayDangleDeps;
  std::unique_ptr<DanglingDeps_t> DanglingDeps;
  std::unique_ptr<RisingDanglingDeps_t> RisingDanglingDeps;
  std::unique_ptr<MayDangleDanglingDeps_t> MayDangleDanglingDeps;
  std::unique_ptr<MayRiseDeps_t> MayRiseDeps;
  std::unique_ptr<MayRiseRisingDeps_t> MayRiseRisingDeps;
  std::unique_ptr<MayRiseMayDangleDeps_t> MayRiseMayDangleDeps;

  void saveStats();
  bool updateStats();

  struct Stats {
    size_t Intact;
    size_t Rising;
    size_t MayDangle;
    size_t Dangling;
    size_t RisingDangling;
    size_t MayDangleDangling;
    size_t MayRise;
    size_t MayRiseRising;
    size_t MayRiseMayDangle;
  } Stats;

  SmallVector<DC> DCs;
};

class LKMMVerifier {
public:
  //LKMMVerifier();

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:

};

#define SAVE_STAT(STAT) Stats.STAT = STAT##Deps->size();
void LKMMAnnotateDepsPass::LKMMAnnotator::saveStats() {
  FOR_EACH_DEP(SAVE_STAT);
}

bool LKMMAnnotateDepsPass::LKMMAnnotator::updateStats() {
  bool Changed = false;
  
#define CMP_AND_PRINT(STAT) \
  do { \
    if ( size_t Diff = STAT##Deps->size() - Stats.STAT ) { \
      Changed = true; \
      errs() << #STAT << " increased by " << Diff << "\n"; \
    } \
    SAVE_STAT(STAT); \
  } while (0);

  FOR_EACH_DEP(CMP_AND_PRINT);
#undef CMP_AND_PRINT

  return Changed;
}
#undef SAVE_STAT

SmallVector<DC<LKMMAnnotateDeps>> LKMMAnnotateDepsPass::LKMMAnnotator::run(Module &M, ModuleAnalysisManager &AM) {
  AnnotCtx AC;

  for (auto &F : M) {

    // TODO: check?
    if (F.empty())
      continue;

    // Annotate dependencies ending in volatile loads and stores.
    AC.passOne(&F, IntactDeps.get(), RisingDeps.get(), MayDangleDeps.get());
  }

  size_t Depth = 5;
  do {
    Depth--;

    for (auto &F : M) {
      //Annotate dependencies ending in returns.
      if (!F.getAttributes().getRetAttrs().hasAttribute("returns_addr_dep"))
        continue;
      AC.passTwo(&F, DanglingDeps.get(), RisingDanglingDeps.get(), MayDangleDanglingDeps.get());
    }
      
    for (auto &F : M) {
      //Annotate dependencies ending in calls.
      if (!F.hasFnAttribute("calls_addr_dep"))
        continue;
      AC.passThree(&F, MayRiseDeps.get(), MayRiseRisingDeps.get(), MayRiseMayDangleDeps.get());
    }
  } while (updateStats() && Depth);

  AC.merge(5);

  return AC.getResult();
}

PreservedAnalyses LKMMVerifier::run(Module &M, ModuleAnalysisManager &AM) {
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

AnalysisKey LKMMAnnotateDepsPass::Key;

LKMMAnnotateDeps LKMMAnnotateDepsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  using DC = llvm::DC<LKMMAnnotateDeps>;
  auto A = LKMMAnnotator();
  auto Res = A.run(M, AM);
  auto *Tmp = new DC[Res.size()];
  std::copy(Res.begin(), Res.end(), Tmp);
  return LKMMAnnotateDeps(ArrayRef<DC>(Tmp, Res.size()));
}

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

} // namespace llvm
