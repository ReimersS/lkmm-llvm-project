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
#include "llvm/Support/raw_ostream.h"
#include <functional>

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

template<int B, int E, typename C>
void removeDuplicates(std::vector<SegmentID<B,E,C>> &Segments) {
  std::sort(Segments.begin(), Segments.end());
  auto Res = std::unique(Segments.begin(), Segments.end(), SegmentID<B,E,C>::equal);
  Segments.erase(Res, Segments.end());
}

/// Represents a dependency chain link on LLVM IR level. A dep chain link consists of an IR
/// instruction and the corresponding dep chain level.
///
/// This is private to the current search, as values may have no meaning
/// after other optimisation passes
class LKMMSearchPolicy::DCLink : public DCLinkBase {
public:
  DCLink(Instruction *Val, const DCLevel Lvl, int Depth) : DCLinkBase(Val->getDebugLoc()?Val->getDebugLoc().get():nullptr, Lvl, Depth), Val(Val) {}
  ~DCLink() = default;

  DCLink(const DCLink &Other) : DCLinkBase(Other.Loc.get(), Other.Lvl, Other.Depth), Val(Other.Val) {}

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
  DCLink(const LKMMSearchPolicy::DCLink &Other) : DCLinkBase(Other.Loc, Other.Lvl, Other.getDepth()), F(Other.Val->getFunction()), Type(DCLinkType::VALUE) {
    if (Other.isCall()) Type = DCLinkType::CALL;
    if (Other.isRet()) Type = DCLinkType::RETURN;
  }

  bool isCall() const { return Type == DCLinkType::CALL; }
  bool isRet() const { return Type == DCLinkType::RETURN; }

  bool operator==(const DCLinkBase &Other) const override {
    const auto &O = static_cast<const DCLink &>(Other);
    return Loc->getLine() == O.Loc->getLine() && Loc.getCol() == O.Loc.getCol() && Depth == O.Depth;
  }

  // Ok to keep pointers to functions.
  // a) Only needed to annotate the IR chains (so definetely valid)
  // b) Optimizations are unlikely to delete entire functions
  const Function *F;
private:
  DCLinkType Type;
};

template<typename Context>
void DC<Context>::addLink(const typename Context::DCLink &Link, std::optional<int> Arg) {

  // In source level chains, we only add Links with a location.
  // This can happen when declaring local variables.
  if constexpr (std::is_same_v<Context, LKMMSearchPolicy>) {
    if (!Link.Loc)
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

/// Convenience specialization.
/// Adds a value to the IR level dependency chain.
template<>
void DC<LKMMSearchPolicy>::addLink(Instruction *Val, DCLevel Lvl, std::optional<int> Arg) {
  LKMMSearchPolicy::DCLink Link(Val, Lvl, 0);
  addLink(Link, Arg);
}

// No need to check for compatibility, the segment specialization
// should ensure this.
/// Concatenates two IR level dependency chains.
/// Merging must be done before annotation, otherwise we lose access to the instructions.
template<>
DC<LKMMSearchPolicy>::DC(const DC &Beg, const DC &End, int Delta) {

  // keep in mind that the chains are in reverse order
  auto It = Chain.insert(Chain.begin(), End.Chain.begin(), End.Chain.end());
  // Chain: [End.E, ...., End.B]
  if (Delta > 0) {
    for (auto &I = It; I != Chain.end(); I++) {
      I->addDepth(Delta);
    }
  }
  It = Chain.insert(Chain.end(), Beg.Chain.begin(), Beg.Chain.end());
  // Chain: [End.E, ...., End.B, Beg.E, ...., Beg.B]
  if (Delta < 0) {
    for (auto &I = It; I != Chain.end(); I++) {
      I->addDepth(-Delta);
    }
  }
  ArgB = Beg.ArgB;
  ArgE = End.ArgE;
}

/// Convenience specialization.
/// One-way copy constructor from an IR level chain, to a source level chain.
template<>
template<>
DC<LKMMAnnotateDeps>::DC(const DC<LKMMSearchPolicy> &Other) {
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
template<> const std::string SegmentID<0, 0, LKMMAnnotateDeps>::Type = "Intact";
template<> const std::string SegmentID<-1, 0, LKMMAnnotateDeps>::Type = "Rising";
template<> const std::string SegmentID<1, 0, LKMMAnnotateDeps>::Type = "May Dangle";
template<> const std::string SegmentID<0, -1, LKMMAnnotateDeps>::Type = "Dangling";
template<> const std::string SegmentID<-1, -1, LKMMAnnotateDeps>::Type = "Rising & Dangling";
template<> const std::string SegmentID<1, -1, LKMMAnnotateDeps>::Type = "May Dangle & Dangling";
template<> const std::string SegmentID<0, 1, LKMMAnnotateDeps>::Type = "May Rise";
template<> const std::string SegmentID<-1, 1, LKMMAnnotateDeps>::Type = "May Rise & Rising";
template<> const std::string SegmentID<1, 1, LKMMAnnotateDeps>::Type = "May Rise & May Dangle";

// Try to find dependencies bottom-up.

class BUCtx : public InstVisitor<BUCtx> {
public:
  enum CtxKind { CK_Annot, CK_Ver };

  CtxKind getKind() const { return Kind; }

  BUCtx(CtxKind CK)
      : Kind(CK){};

  void runSearch() {
    for (auto &BB : *F) {
      visitBasicBlock(BB);
    }
  }

  // Generic forwarder for all values.
  void visit(Value *V);

  // Helper to add branches to chain
  void handleBranch(BasicBlock *NextBB);

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

  // FIXME: is this "conditional"?
  void visitPHINode(PHINode &PN) {};

  void visitTruncInst(TruncInst &TI) {};

  void visitZExtInst(ZExtInst &ZI) {};

  void visitSExtInst(SExtInst &SI) {};

  void visitPtrToIntInst(PtrToIntInst &PTI) {};

  void visitIntToPtrInst(IntToPtrInst &ITPI) {};

  void visitBitCastInst(BitCastInst &BCI) {};

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASCI) {};

  void visitSelectInst(SelectInst &SI);

protected:
  // The function the BFS is currently visiting.
  Function *F;

  // The BB the BFS is currently checking.
  BasicBlock *BB;

private:
  const CtxKind Kind;
};

class LKMMSearchPolicy::AnnotCtx : public BUCtx {
public:

  AnnotCtx(BUCtx::CtxKind Ctx,
      void (* AnnoFn)(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result),
      LKMMAnnotateDeps *PrevResult) : BUCtx(Ctx), Result(new(llvm::LKMMAnnotateDeps::DepMap)), CurrPass(Pass::Known_End), AnnotateFn(AnnoFn), PrevResult(PrevResult){};

  void setNewDc(std::unique_ptr<DC> NewDC) {
    CurrDC = std::move(NewDC);
  }

  std::unique_ptr<DC> getDCPtr() {
    return std::move(CurrDC);
  }

  DC &getDc() {
    return *CurrDC;
  }

  auto getResult() { return Result; }

  // Only runs once. Annotates ALL segments ending in volatile loads and stores.
  void passOne(Function *NewF, IntactDeps_t *I, RisingDeps_t *R, MayDangleDeps_t *MD) {
    F = NewF;
    this->I = I;
    this->R = R;
    this->MD = MD;

    runSearch();
  }

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

    removeDuplicates(*Result);
    for (auto I : *Result) {
      I.print();
    }
  }

  template<int B, int E>
  void makeIntactDep() {

    DepMap<B, E> *TypedMap;

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

      auto *Callee = cast<CallInst>(CurrDC->Chain.front().Val);
      if (Callee->getCalledFunction()) {
        assert(Callee->getCalledFunction()->getAttributes().getFnAttrs().hasAttribute("takes_addr_dep") && "Function should not have been passed in Pass 3");
        assert(Callee->getCalledFunction()->getAttributes().getParamAttrs(CurrDC->ArgE.value()).hasAttribute("is_addr_dep") && "Argument should not have been passed in Pass 3");
      }
    }

    // FIXME
    auto Seg = SegmentID<B, E, LKMMSearchPolicy>(getDc());
    getDCPtr().release();
    TypedMap->push_back(Seg);
  }

  enum Pass { Known_End, Known_Ret, Known_Call, Match };

  Pass currPass() const { return CurrPass; }

private:
  // Currently tracked DC.
  std::unique_ptr<DC> CurrDC;

  llvm::LKMMAnnotateDeps::DepMap *Result;

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

  void (*AnnotateFn)(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result);

  LKMMAnnotateDeps *PrevResult;

  void buildTransitiveClosure(const size_t Depth);
  template<int B, int M, int E>
  DepMap<B,E> match(DepMap<B, M> *Beg, DepMap<-M, E> *End);
};

/// Converts an LLVM IR level chain to a source level chain, and
/// annotates the chain in the IR.
void LKMMAnnotateDepsPass::annotateChain(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result) {

    std::string Annot;
    std::string Pretty;

    llvm::SegmentID<0,0,LKMMAnnotateDeps> Ret(Seg);

    for (auto I = Ret.getDC().Chain.crbegin(); I != Ret.getDC().Chain.crend(); I++) {
      Annot += getInstLocString(I->F->getName(), I->Loc);
      Pretty += std::string(I->getDepth(), '\t') + getInstLocString(I->F->getName(), I->Loc);
      if (I != std::prev(Ret.getDC().Chain.crend())) {
        Annot += "--";
        Pretty += "\n";
      }
    }

    {
      auto I = Seg.getDC().Chain.back();
      MDNode *Meta = MDNode::get(I.Val->getContext(), MDString::get(I.Val->getContext(), Annot));

      if (auto *Existing = I.Val->getMetadata("addr_dep"))
        Meta = llvm::MDNode::concatenate(Existing, Meta);
      I.Val->setMetadata("addr_dep", Meta);
    }
    {
      auto I = Seg.getDC().Chain.front();
      MDNode *Meta = MDNode::get(I.Val->getContext(), MDString::get(I.Val->getContext(), Annot));

      if (auto *Existing = I.Val->getMetadata("addr_dep"))
        Meta = llvm::MDNode::concatenate(Existing, Meta);
      I.Val->setMetadata("addr_dep", Meta);
    }

    Ret.setStr(Pretty);
    Result->push_back(Ret);
}

/// Converts an LLVM IR level chain to a source level chain, and
/// optionally checks for annotations from the previous rounds.
void LKMMVerifyDepsPass::addChain(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result) {
  std::string Annot;
  std::string Pretty;

  llvm::SegmentID<0,0,LKMMAnnotateDeps> Ret(Seg);

  for (auto I = Ret.getDC().Chain.crbegin(); I != Ret.getDC().Chain.crend(); I++) {
    Annot += getInstLocString(I->F->getName(), I->Loc);
    Pretty += std::string(I->getDepth(), '\t') + getInstLocString(I->F->getName(), I->Loc);
    if (I != std::prev(Ret.getDC().Chain.crend())) {
      Annot += "--";
      Pretty += "\n";
    }
  }

  /*
  for (auto I = Seg.getDC().Chain.crbegin(); I != Seg.getDC().Chain.crend(); I++) {
    MDNode *Meta = MDNode::get(I->Val->getContext(), MDString::get(I->Val->getContext(), Annot));

    if (auto *Existing = I->Val->getMetadata("addr_dep"))
      Meta = llvm::MDNode::concatenate(Existing, Meta);
    I->Val->setMetadata("addr_dep", Meta);
  }
  */

  Ret.setStr(Pretty);

  Result->push_back(Ret);
}

template<int B, int M, int E>
LKMMSearchPolicy::DepMap<B,E> LKMMSearchPolicy::AnnotCtx::match(DepMap<B, M> *Beg, DepMap<-M, E> *End) {
  if (!Beg || !End)
    return DepMap<B,E>();

  DepMap<B,E> Ret;
  for (auto &EndSeg : *End) {
    for (auto &BegSeg : *Beg) {
      if (BegSeg.isCompatible(EndSeg)) {
        Ret.push_back(SegmentID<B,E,LKMMSearchPolicy>(BegSeg, EndSeg));
      }
    }
  }

  removeDuplicates(Ret);
  return Ret;
}

void LKMMSearchPolicy::AnnotCtx::buildTransitiveClosure(const size_t Depth) {

  LKMMAnnotateDeps::DepMap *IntactDeps = Result;

  if (Depth == 1) {
    // Depth 1: Trivial
    for (auto &Seg : *I) {
      AnnotateFn(Seg, IntactDeps);
    }
    return;
  }

  if (Depth == 2) {
    // Depth 2: Still trivial, concatenate all matching R/MR and MD/D pairs
    auto DCs = match(MR, R);
    for (auto &Seg : DCs) {
      AnnotateFn(Seg, IntactDeps);
    }

    DCs = match(D, MD);
    for (auto &Seg : DCs) {
      AnnotateFn(Seg, IntactDeps);
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

  for (auto &Seg : match(MRR, R)) {
    PrevNeg->push_back(Seg);
  }
  for (auto &Seg : match(MRMD, R)) {
    PrevPos->push_back(Seg);
  }
  for (auto &Seg : match(MDD, MD)) {
    PrevPos->push_back(Seg);
  }
  for (auto &Seg : match(RD, MD)) {
    PrevNeg->push_back(Seg);
  }

  // sort & remove duplicates ??

  // TODO: check for delta
  while (Len <= Depth-1) {
    for (auto &Seg : match(MRR, PrevNeg.get())) {
      CurNeg->push_back(Seg);
    }
    for (auto &Seg : match(MRMD, PrevNeg.get())) {
      CurPos->push_back(Seg);
    }
    for (auto &Seg : match(MDD, PrevPos.get())) {
      CurPos->push_back(Seg);
    }
    for (auto &Seg : match(RD, PrevPos.get())) {
      CurNeg->push_back(Seg);
    }

    PrevNeg = std::move(CurNeg);
    PrevPos = std::move(CurPos);
    CurNeg = std::make_unique<DepMap<-1, 0>>();
    CurPos = std::make_unique<DepMap<1, 0>>();
    Len++;
  }

  for (auto &Seg : match(MR, PrevNeg.get())) {
      AnnotateFn(Seg, IntactDeps);
  }
  for (auto &Seg : match(D, PrevPos.get())) {
      AnnotateFn(Seg, IntactDeps);
  }
}

void BUCtx::handleBranch(BasicBlock *NextBB) {
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;
  auto *CurrBB = Ann->getDc().Chain.back().Val->getParent();

  if (auto *BI = dyn_cast<BranchInst>(NextBB->getTerminator())) {
    if (!BI->isConditional())
      return;
    // NOTE: This is one case where we could get false positives.
    // We don't retroactively check if any path breaks the chain,
    // we just know at least one doesn't
    //
    // Example:
    //                 +---NextBB---+
    //                 |     br     | visit(inst) <----+
    //                 +------------+                  |
    //                     /    \                      |
    //            +-------+      +-------+             |
    //            | EMPTY |      | BREAK |         depends on
    //            +-------+      +-------+             |
    //                    \      /                     |
    //                 +---CurrBB---+                  |
    //                 |            | chain.back()-----+
    //                 +------------+
    //

    // If there is a trivial path from one target
    // to the current BB, we add the branch.

    for (auto *Succ : BI->successors()) {
      if (!isPotentiallyReachable(Succ, CurrBB))
        continue;
      auto *T = Succ;
      while (T != CurrBB) {
        T = T->getUniqueSuccessor();
        if (!T)
          return;
      }
      Ann->getDc().addLink(BI, DCLevel::EMPTY);
      // There should be exactly one terminator for the next BB
      return;
    }
    return;
  }
}

void BUCtx::visit(Value *V) {
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;
  if (auto *I = dyn_cast<Instruction>(V)) {
    auto *NextBB = I->getParent();
    // Whatever we want to insert, it is in a different BB.
    // We need to check for branches, even for addr & data dependencies
    // because they can be optimized to selects.
    if (NextBB != Ann->getDc().Chain.back().Val->getParent()) {
      handleBranch(NextBB);
    }
    InstVisitor::visit(I);
  }
  if (auto *A = dyn_cast<Argument>(V)) {
    visitArgument(A);
  }
}

// TODO: avoid back edges
void BUCtx::visitBasicBlock(BasicBlock &BB) {
  this->BB = &BB;
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  if (Ann->currPass() == LKMMSearchPolicy::AnnotCtx::Pass::Known_End) {
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

          auto End = std::make_unique<DC<LKMMSearchPolicy>>();
          End->addLink(&I, DCLevel::PTR);
          Ann->setNewDc(std::move(End));
          visit(Ptr);
      }
    }
      return;
  } // !Known_End

  if (Ann->currPass() == LKMMSearchPolicy::AnnotCtx::Pass::Known_Ret) {
    // We also need to track any potential chains from return values.
    // FIXME: Aggregate returns should have an annotation per element.
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      for (auto &Op : RI->operands()) {
        auto End = std::make_unique<DC<LKMMSearchPolicy>>();
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

  if (Ann->currPass() == LKMMSearchPolicy::AnnotCtx::Pass::Known_Call) {
    for (auto &I : BB) {
    // We also need to track any potential chains from call isntructions.
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        // Only track calls to functions we have seen before.
        if (CI->getCalledFunction() && !CI->getCalledFunction()->getAttributes().hasFnAttr("takes_addr_dep"))
          continue;

        for (auto &Arg : CI->args()) {
          auto ArgNo = CI->getArgOperandNo(&Arg);
          if (CI->getCalledFunction() && !CI->getCalledFunction()->getAttributes().hasParamAttr(ArgNo, "is_addr_dep"))
              continue;

          auto End = std::make_unique<DC<LKMMSearchPolicy>>();
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
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  // We found a rising segment!
  // Add a new segment for all call sites of F (likely outside of F)

  for (auto *CallingInstr : F->users()) {
    if (auto *CI = dyn_cast<CallInst>(CallingInstr)) {

      auto Curr = Ann->getDCPtr();
      auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);

      Ann->setNewDc(std::move(Cpy));
      Ann->getDc().addLink(CI, getLastNonEmptyLvl(Curr->Chain), A->getArgNo());

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
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  for (auto L = Ann->getDc().Chain.crbegin(); L != Ann->getDc().Chain.crend(); L++) {
    if (L->Lvl == DCLevel::PTE)
      break;
    if (L->Lvl == DCLevel::EMPTY)
      continue;
    return;
  }
  Ann->getDc().addLink(&SI, DCLevel::PTR);
  auto *Val = SI.getValueOperand();
  visit(Val);
}

void BUCtx::visitLoad(LoadInst &LI) {
  if (LI.isVolatile()) {
    // We found an internal beginning!
    auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

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

void BUCtx::goThroughMem(LoadInst &LI) {

  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  // TODO: Are double load/stores ok?
  // Sounds like aliasing
  if (getLastNonEmptyLvl(Ann->getDc().Chain) == DCLevel::PTE) return;

  assert(getLastNonEmptyLvl(Ann->getDc().Chain) == DCLevel::PTR &&
         "Expected a pointer to be the last link in the chain for Load");
  Ann->getDc().addLink(&LI, DCLevel::PTE);

  // Find previous stores that write the same location and continue there.
  // Anything else is potential alias territory (conservatively speaking)
  for (auto *U : LI.getPointerOperand()->users()) {
    if (!isa<StoreInst>(U))
      continue;
    auto *SI = cast<StoreInst>(U);

    if (!isPotentiallyReachable(SI->getParent(), LI.getParent())) continue;

    for (auto L = Ann->getDc().Chain.crbegin(); L != Ann->getDc().Chain.crend(); L++) {
      if (L->Val == SI)
        goto skip;
    }
    {
      if (SI->getPointerOperand() == LI.getPointerOperand()) {
        auto Curr = Ann->getDCPtr();
        auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);

        Ann->setNewDc(std::move(Cpy));
        visit(SI);
        Ann->setNewDc(std::move(Curr));
      }
    }
skip:
    ;
  }
}

void BUCtx::visitGetElementPtrInst(GetElementPtrInst &GEP) {

  // GEP is a glorified add
  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  assert(getLastNonEmptyLvl(Ann->getDc().Chain) == DCLevel::PTR &&
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

  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;

  // We found segments that may dangle!
  // Add a new segment for all returns in the callee.

  Ann->getDc().addLink(&CI, getLastNonEmptyLvl(Ann->getDc().Chain));

  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return;

  for (auto &BB : *Callee) {
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      auto Curr = Ann->getDCPtr();
      auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);

      Ann->setNewDc(std::move(Cpy));
      Ann->getDc().addLink(RI, getLastNonEmptyLvl(Curr->Chain));

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

void BUCtx::visitSelectInst(SelectInst &SI) {

  auto *Ann = (LKMMSearchPolicy::AnnotCtx *)this;
  Ann->getDc().addLink(&SI, getLastNonEmptyLvl(Ann->getDc().Chain));

  for (auto &Op : SI.operands()) {
    auto Curr = Ann->getDCPtr();
    auto Cpy = std::make_unique<decltype(Curr)::element_type>(*Curr);

    Ann->setNewDc(std::move(Cpy));
    visit(Op);
    Ann->setNewDc(std::move(Curr));
  }
}

class LKMMSearchPolicy::LKMMAnnotator {
public:
  LKMMAnnotator( BUCtx::CtxKind Kind, void (* AnnoFn)(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result), LKMMAnnotateDeps *PrevResult = nullptr) :
                    Kind(Kind),
                    AnnoFn(AnnoFn),
                    PrevResult(PrevResult),
                    IntactDeps(std::make_unique<IntactDeps_t>()),
                    RisingDeps(std::make_unique<RisingDeps_t>()),
                    MayDangleDeps(std::make_unique<MayDangleDeps_t>()),
                    DanglingDeps(std::make_unique<DanglingDeps_t>()),
                    RisingDanglingDeps(std::make_unique<RisingDanglingDeps_t>()),
                    MayDangleDanglingDeps(std::make_unique<MayDangleDanglingDeps_t>()),
                    MayRiseDeps(std::make_unique<MayRiseDeps_t>()),
                    MayRiseRisingDeps(std::make_unique<MayRiseRisingDeps_t>()),
                    MayRiseMayDangleDeps(std::make_unique<MayRiseMayDangleDeps_t>()),
                    Stats({}) {};

  llvm::LKMMAnnotateDeps::DepMap *run(Module &M, ModuleAnalysisManager &AM);

private:
  BUCtx::CtxKind Kind;
  void (* AnnoFn)(const SegmentID<0,0, LKMMSearchPolicy> &Seg, LKMMAnnotateDeps::DepMap *Result);
  llvm::LKMMAnnotateDeps *PrevResult;

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
void LKMMSearchPolicy::LKMMAnnotator::saveStats() {
  FOR_EACH_DEP(SAVE_STAT);
}

bool LKMMSearchPolicy::LKMMAnnotator::updateStats() {
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

llvm::LKMMAnnotateDeps::DepMap *LKMMSearchPolicy::LKMMAnnotator::run(Module &M, ModuleAnalysisManager &AM) {
  AnnotCtx AC(Kind, AnnoFn, PrevResult);

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

  // only merge if at least one pass ran
  auto *Ret = AC.getResult();
  if (Ret->empty())
    return Ret;

  AC.merge(5);
  return AC.getResult();
}

PreservedAnalyses LKMMVerifier::run(Module &M, ModuleAnalysisManager &AM) {
  return PreservedAnalyses::all();
}

void LKMMVerifyDepsPass::verifyChain(LKMMAnnotateDeps::DepMap *Pre, LKMMAnnotateDeps::DepMap *Post) {

  for (auto &Seg : *Pre) {

    auto It = std::find_if(Post->begin(), Post->end(),
        [&Seg](const SegmentID<0,0,LKMMAnnotateDeps> &PostSeg) {

      // Compare beggingin and end first
      return Seg == PostSeg;
    });

    if (It == Post->end()) {
      llvm::errs() << "Missing chain for: " << Seg.Pretty << "\n\n";
      return;
    }

    auto Dbg = It;

    // The segments should be sorted
    while (Seg == *It) {
      for (auto Link = It->getDC().Chain.cbegin(); Link != It->getDC().Chain.cend(); Link++) {
        auto Jt = std::find_if(Seg.getDC().Chain.cbegin(), Seg.getDC().Chain.cend(),
          [&Link](const LKMMAnnotateDeps::DCLink &SegLink) {
          return Link->Loc == SegLink.Loc;
        });
        if (Jt == Seg.getDC().Chain.cend()) {
          llvm::errs() << "Missing Link: " << getInstLocString(Link->F->getName(), Link->Loc, false) << "\n\n";
          break;
        }
      }
      It++;
    }

  //#ifdef LLVM_DEBUG
    errs() << "\nMatched:\nPRE-OPT:\n" <<Seg.Pretty <<
    "\nPOST_OPT:\n" << Dbg->Pretty << "\n\n";
  //#endif
  }
}

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

AnalysisKey LKMMAnnotateDepsPass::Key;

LKMMAnnotateDeps LKMMAnnotateDepsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto A = LKMMSearchPolicy::LKMMAnnotator(BUCtx::CtxKind::CK_Annot, &annotateChain);
  return LKMMAnnotateDeps(A.run(M, AM));
}

bool LKMMAnnotateDeps::invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<LKMMAnnotateDepsPass>();
  return !PAC.preservedWhenStateless();
}

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//
PreservedAnalyses LKMMVerifyDepsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &Annotations = AM.getResult<LKMMAnnotateDepsPass>(M);
  errs() << "\nvvvvv~~~~~~~~~ LKMMVerifyDepsPass ~~~~~vvvvv\n";
  auto A = LKMMSearchPolicy::LKMMAnnotator(BUCtx::CtxKind::CK_Ver, &addChain, &Annotations);
  verifyChain(Annotations.IntactDeps, A.run(M, AM));
  errs() << "\n^^^^^~~~~~~~~~ LKMMVerifyDepsPass ~~~~~^^^^^\n";

  return PreservedAnalyses::all();
}

} // namespace llvm
