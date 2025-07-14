//===- llvm/Transforms/LKMMDependenceAnalysis.h - LKMM Deps -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains all declarations / definitions required for LKMM
/// dependence analysis. Implementations live in LKMMDependenceAnalysis.cpp.
///
//===----------------------------------------------------------------------===//

#include <concepts>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_set>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/PassManager.h"

#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H

namespace llvm {
  class LKMMAnnotateDeps;

//===----------------------------------------------------------------------===//
// Some common types
//===----------------------------------------------------------------------===//

// TODO: Reduce
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

enum Reason { COMPLETE, EXTERN, OVERWRITE, DOUBLE_MEM };

enum DepType { ADDR, DATA, CTRL };

enum CtxKind { CK_Annot, CK_Ver };

class DCLinkBase {
public:
  DCLinkBase(DILocation *Loc, DCLevel Lvl, int Depth)
      : Loc(Loc), Lvl(Lvl), Depth(Depth) {}

  //DCLinkBase(const DCLinkBase &Other) : Loc(Other.Loc), Lvl(Other.Lvl), Depth(Other.Depth) {}
  virtual ~DCLinkBase() = default;

  DebugLoc Loc;
  DCLevel Lvl;

  bool isCall() const;
  bool isRet() const;

  void addDepth(int Delta) { Depth += Delta; }
  int getDepth() const { return Depth; }

  virtual bool operator==(const DCLinkBase &Other) const = 0;

protected:
  int Depth;
};

/// Represents a dependency chain (segment). A dep chain consists of a beginning, an
/// ending, and a unique chain of links between them.
///
/// We use the names "dependency chain" and "chain segment" interchangeably.
template <typename Context>
struct DC {
  DC() {};

  template <typename T>
  DC<Context>(const DC<T> &Other) = delete;

  DC(const DC &A, const DC &B, int Delta) = delete;

  bool addLink(const typename Context::DCLink &Link, std::optional<int> Arg = std::nullopt);

  bool addLink(Instruction *Val, DCLevel Lvl, std::optional<int> Arg = std::nullopt) = delete;

  bool insertLink(Instruction *Val, DCLevel Lvl, std::optional<int> Arg = std::nullopt) = delete;

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

  bool operator<(const DC &Other) const {

    if (Chain.size() != Other.Chain.size())
      return Chain.size() < Other.Chain.size();

    for (size_t i = 0; i < Chain.size(); i++) {
      const auto &L = Chain[i];
      const auto &R = Other.Chain[i];

      if (L.Loc.getLine() != R.Loc.getLine())
        return L.Loc.getLine() < R.Loc.getLine();
      if (L.Loc.getCol() != R.Loc.getCol())
        return L.Loc.getCol() < R.Loc.getCol();
    }

    if (ArgB != Other.ArgB)
      return ArgB < Other.ArgB;

    if (ArgE != Other.ArgE)
      return ArgE < Other.ArgE;

    return false; // equal
  }

  bool operator==(const DC &Other) const {
    return !(*this < Other) && !(Other < *this);
  }

  class DCHash {
  public:
    std::size_t operator()(const DC &DC) const noexcept {
      return hash_combine(std::hash<decltype(DC.Chain)>{}(DC.Chain), DC.ArgB.value_or(0), DC.ArgE.value_or(0));
    }
  };
};

template <int B, int E>
struct SegmentType {
  static constexpr std::string_view Type;
};

template<> constexpr std::string_view SegmentType<0, 0>::Type = "Intact";
template<> constexpr std::string_view SegmentType<-1, 0>::Type = "Rising";
template<> constexpr std::string_view SegmentType<1, 0>::Type = "May Dangle";
template<> constexpr std::string_view SegmentType<0, -1>::Type = "Dangling";
template<> constexpr std::string_view SegmentType<-1, -1>::Type = "Rising & Dangling";
template<> constexpr std::string_view SegmentType<1, -1>::Type = "May Dangle & Dangling";
template<> constexpr std::string_view SegmentType<0, 1>::Type = "May Rise";
template<> constexpr std::string_view SegmentType<-1, 1>::Type = "May Rise & Rising";
template<> constexpr std::string_view SegmentType<1, 1>::Type = "May Rise & May Dangle";

template <int B, int E, typename C>
class SegmentID {
  // For each segment we need to store:
  // the Pair, the Function, the Arg numbers,
public:
  SegmentID(DC<C> &Dc) : Dc(Dc) {

    // Will throw if we messed up
    if constexpr (B == -1) {
      assert(Dc.ArgB.has_value());
    }
    if constexpr (E == 1) {
      assert(Dc.ArgE.has_value());
    }
  }

  template<typename O>
  SegmentID(const SegmentID<B,E,O> &Other) : Pretty(Other.Pretty), Dc(Other.getDC()) {}

  //SegmentID(SegmentID<B,E,C> &Other) : Dc(Other.getDC()) {}

  //SegmentID& operator=(const SegmentID<B,E,C> &Other) { Dc = Other.getDC(); return *this; }

  template<int M>
  SegmentID(SegmentID<B,M,C> &Beg, SegmentID<-M,E,C> &End) : Dc(Beg.getDC(), End.getDC(), M) {}

  /// Returns true if [*this*, Other] is a valid segment.
  template<int BO, int EO>
  bool isCompatible(const SegmentID<BO,EO,C> &Other) const {
    // Segments must match at function boundaries
    if constexpr (E == 0 | BO == 0)
      return false;

    // Rising matches MayRise, Dangling matches MayDangle
    if constexpr (E + BO != 0)
      return false;

    // MR/R meet at call instructions, arguments must match
    if constexpr (E == 1) {
      if (this->getArgE() != Other.getArgB())
        return false;

      return this->getEnd() == Other.getBegin();
    }

    // D/MD meet at return instructions
    if constexpr (E == -1) {
      return this->getEnd() == Other.getBegin();
    }

    llvm_unreachable("Invalid segment combination");
  }

  // To sort
  // (1) Lex. by filename
  // (2) by earliest end loc
  // (3) by earliest beg loc
  // Since we search bottom up, the annotator will insert the segments in roughly this order
  bool operator<(const SegmentID &Other) const {
    auto LEnd = this->getEnd();
    auto REnd = Other.getEnd();

    if (LEnd != REnd) {
      if (!LEnd)
        return true;
      if (!REnd)
        return false;

      auto *LScope = cast_or_null<DIScope>(LEnd->getScope());
      auto *RScope = cast_or_null<DIScope>(REnd->getScope());

      if (LScope != RScope) {
        if (!LScope)
          return true;
        if (!RScope)
          return false;

        if (LScope->getFilename() != RScope->getFilename())
          return LScope->getFilename() < RScope->getFilename();
      }

      if (LEnd.getLine() != REnd.getLine())
        return LEnd.getLine() < REnd.getLine();

      if (LEnd.getCol() != REnd.getCol())
        return LEnd.getCol() < REnd.getCol();
    }

    auto LBegin = this->getBegin();
    auto RBegin = Other.getBegin();

    if (LBegin != RBegin) {
      if (!LBegin)
        return true;
      if (!RBegin)
        return false;

      if (LBegin.getLine() != RBegin.getLine())
        return LBegin.getLine() < RBegin.getLine();

      if (LBegin.getCol() != RBegin.getCol())
        return LBegin.getCol() < RBegin.getCol();
    }

    return false;
  }

  bool operator==(const SegmentID &Other) const {
    return !(*this < Other) && !(Other < *this);
  }


  // To remove duplicates
  static bool equal(const SegmentID &LHS, const SegmentID &RHS) {
    if (LHS == RHS)
      return LHS.Dc == RHS.Dc;
    return false;
  }

  static bool lt(const SegmentID &LHS, const SegmentID &RHS) {
    if (LHS < RHS)
      return true;
    //if (RHS < LHS)
    //  return false;
    return LHS.Dc < RHS.Dc;
  }

  void makePretty();
  void setStr(const std::string &Str) { Pretty = Str; }
  void print() { errs() << Pretty << "\n\n"; };

  // TODO: string_view?
  const std::string_view &getType() const { return Type; }
  const DebugLoc getBegin() const { return Dc.Chain.back().Loc; }
  const DebugLoc getEnd() const { return Dc.Chain.front().Loc; }
  std::optional<int> getArgB() const { return Dc.ArgB; }
  std::optional<int> getArgE() const { return Dc.ArgE; }
  constexpr int getB() const { return B; }
  constexpr int getE() const { return E; }
  constexpr int delta() { return E - B; }

  const DC<C> &getDC() const { return Dc; }
  static constexpr std::string_view Type = SegmentType<B, E>::Type;
  std::string Pretty;

private:
  DC<C> Dc;
};

//===----------------------------------------------------------------------===//
// Some helper functions
//===----------------------------------------------------------------------===//

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
std::string getInstLocString(Instruction *I, bool ViaFile = false);

std::string getInstLocString(const StringRef &F ,const DebugLoc &InstDebugLoc, bool ViaFile = false);

/// _Sorts_ and removes duplicates from the given vector of segments.
template<int B, int E, typename C>
void removeDuplicates(std::vector<SegmentID<B, E, C>> &Segments);

/// \returns the last non-EMPTY lvl in a \p Chain.
///
/// \param Chain the _non-empty_ chain to search.
template <typename C>
DCLevel getLastNonEmptyLvl(std::vector<C> &Chain) {
  for (auto It = Chain.rbegin(); It != Chain.rend(); ++It) {
    if (It->Lvl != DCLevel::EMPTY)
      return It->Lvl;
  }

  //llvm_unreachable("Chain is empty, no non-EMPTY level found");
  return DCLevel::EMPTY; //FIXME
}

//===----------------------------------------------------------------------===//
// The Dependency Analysis
//===----------------------------------------------------------------------===//

class LKMMAnnotateDeps {

public:
  using DC = DC<LKMMAnnotateDeps>;
  using DepMap = std::vector<SegmentID<0,0, LKMMAnnotateDeps>>;

  enum DCLinkType { VALUE, CALL, RETURN };
  class DCLink;

  DepMap *IntactDeps[3];

  // never invalidate this
  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &);

  void add(DepType DT, DepMap *Result) { IntactDeps[(int)DT] = Result; };
};

//===----------------------------------------------------------------------===//
// The IR search
//===----------------------------------------------------------------------===//
class LKMMSearchPolicy {
public:
  using DC = DC<LKMMSearchPolicy>;

  template<int B, int E>
  using DepMap = std::vector<SegmentID<B,E, LKMMSearchPolicy>>;

  typedef DepMap<0,0> IntactDeps_t;
  typedef DepMap<-1,0> RisingDeps_t;
  typedef DepMap<1,0> MayDangleDeps_t;
  typedef DepMap<0,-1> DanglingDeps_t;
  typedef DepMap<-1,-1> RisingDanglingDeps_t;
  typedef DepMap<1,-1> MayDangleDanglingDeps_t;
  typedef DepMap<0,1> MayRiseDeps_t;
  typedef DepMap<-1,1> MayRiseRisingDeps_t;
  typedef DepMap<1,1> MayRiseMayDangleDeps_t;

  //std::unique_ptr<llvm::DC<LKMMAnnotateDeps> *> DCs;
  class DCLink;

  template<DepType DT>
  class AnnotCtx;

  class LKMMAnnotator;
};

//===----------------------------------------------------------------------===//
// The Actual Annotation Pass
//===----------------------------------------------------------------------===//

class LKMMAnnotateDepsPass : public AnalysisInfoMixin<LKMMAnnotateDepsPass> {
public:
  static AnalysisKey Key;
  friend AnalysisInfoMixin<LKMMAnnotateDepsPass>;

  typedef LKMMAnnotateDeps Result;
  Result run(Module &M, ModuleAnalysisManager &AM);

private:
  LKMMSearchPolicy Policy;
  static void annotateChain(const SegmentID<0,0, LKMMSearchPolicy> &Seg, const DepType DT, LKMMAnnotateDeps::DepMap *Result = nullptr);
};

//===----------------------------------------------------------------------===//
// The Hook Pass
//===----------------------------------------------------------------------===//

/// A wrapper around LKMMAnnotateDepsPass, that is able to be inserted into
/// the earliest hook point.
class LKMMAnnotateHook : public PassInfoMixin<LKMMAnnotateHook> {
public:

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {

    errs() << "\nvvvvv~~~~~~~~~ LKMMAnnotateHook ~~~~~vvvvv\n";
    auto &Annotations = AM.getResult<LKMMAnnotateDepsPass>(M);
    errs() << "\n^^^^^~~~~~~~~~ LKMMAnnotateHook ~~~~~^^^^^\n";
    return PreservedAnalyses::all();
  }
};

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

class LKMMVerifyDepsPass : public PassInfoMixin<LKMMVerifyDepsPass> {
public:
  typedef LKMMAnnotateDeps Result;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  LKMMSearchPolicy Policy;
  void verifyChain(LKMMAnnotateDeps::DepMap *Pre, LKMMAnnotateDeps::DepMap *Post, Module &M);
  static void addChain(const SegmentID<0,0, LKMMSearchPolicy> &Seg, const DepType DT, LKMMAnnotateDeps::DepMap *Result);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CUSTOMMEMDEP_H
