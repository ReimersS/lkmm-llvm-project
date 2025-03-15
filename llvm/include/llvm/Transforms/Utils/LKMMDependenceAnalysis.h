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
#include <type_traits>
#include <unordered_set>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/PassManager.h"

#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H

namespace llvm {

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

class DCLinkBase {
public:
  DCLinkBase(const DebugLoc *Loc, DCLevel Lvl, int Depth = 0)
      : Loc(Loc), Lvl(Lvl), Depth(Depth) {}

  virtual ~DCLinkBase() = default;

  const DebugLoc *Loc;
  DCLevel Lvl;
  
  bool isCall() const;
  bool isRet() const;
  
  void addDepth(int Delta) { Depth += Delta; }
  int getDepth() const { return (Depth > 0) ? Depth : 0; }

  virtual bool operator==(const DCLinkBase &Other) const = 0;

protected:
  int Depth;
};

template <typename Context>
struct DC;

template <int B, int E>
class SegmentID {
  // For each segment we need to store: 
  // the Pair, the Function, the Arg numbers,
public:
  template<typename T>
  SegmentID(DC<T> &Dc) : Begin(Dc.Chain.back().Loc), End(Dc.Chain.front().Loc) {

    // Will throw if we messed up
    if constexpr (B == -1) {
      ArgB = Dc.ArgB.value();
    }
    if constexpr (E == 1) {
      ArgE = Dc.ArgE.value();
    }
  }

  SegmentID(const SegmentID &Other)
    : Begin(Other.Begin), End(Other.End), ArgB(Other.ArgB), ArgE(Other.ArgE) {}

  /// Returns true if [*this*, Other] is a valid segment.   
  template<int BO, int EO>
  bool isCompatible(const SegmentID<BO,EO> &Other) const {
    // Segments must match at function boundaries
    if constexpr (E == 0 | BO == 0)
      return false;

    // Rising matches MayRise, Dangling matches MayDangle
    if constexpr (E + BO != 0)
      return false;

    // MR/R meet at call instructions, arguments must match
    if constexpr (E == 1) {
      if (ArgE != Other.getArgB())
        return false;
      
      return *End == *Other.getBegin();
    }

    // D/MD meet at return instructions
    if constexpr (E == -1) {
      return *End == *Other.getBegin();
    }

    llvm_unreachable("Invalid segment combination");
  }

  // TODO: C++20
  bool operator==(const SegmentID &Other) const {
    return Begin == Other.Begin && End == Other.End &&
           ArgB == Other.ArgB && ArgE == Other.ArgE;
  }

  class SegmentIDHash {
  public:
    std::size_t operator()(const SegmentID &ID) const noexcept {
      // probably good enough
      return
        std::hash<unsigned int>{}(ID.Begin->getCol()) ^
        std::hash<unsigned int>{}(ID.Begin->getLine()) ^
        std::hash<unsigned int>{}(ID.End->getCol() << 4) ^
        std::hash<unsigned int>{}(ID.End->getLine() << 4);
    }
  };


  // TODO: string_view?
  const std::string &getType() const { return Type; }
  const DebugLoc *getBegin() const { return Begin; }
  const DebugLoc *getEnd() const { return End; }
  const std::optional<int> getArgB() const { return ArgB; }
  const std::optional<int> getArgE() const { return ArgE; }
  constexpr int getB() const { return B; }
  constexpr int getE() const { return E; }
  constexpr int delta() { return E - B; }

  static const std::string Type;
private:
  const DebugLoc *Begin;
  const DebugLoc *End;
  std::optional<int> ArgB;
  std::optional<int> ArgE; 
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

//===----------------------------------------------------------------------===//
// The Dependency Analysis
//===----------------------------------------------------------------------===//

class LKMMAnnotateDeps {

public:
  using DC = DC<LKMMAnnotateDeps>;
  LKMMAnnotateDeps(const ArrayRef<DC> &Deps) : IntactDeps(Deps) {}
  
  enum DCLinkType { VALUE, CALL, RETURN };
  class DCLink;

  ArrayRef<DC> IntactDeps;
};

//===----------------------------------------------------------------------===//
// The Actual Annotation Pass
//===----------------------------------------------------------------------===//

class LKMMAnnotateDepsPass : public AnalysisInfoMixin<LKMMAnnotateDepsPass> {
public:
  static AnalysisKey Key;
  friend AnalysisInfoMixin<LKMMAnnotateDepsPass>;

  using DC = DC<LKMMAnnotateDepsPass>;
  typedef LKMMAnnotateDeps Result;
  Result run(Module &M, ModuleAnalysisManager &AM);

  class DCLink;
  class AnnotCtx;
  class LKMMAnnotator;

private:
  template<int B, int E>
  using DepMap = std::unordered_map<
    SegmentID<B,E>,
    std::unordered_set<std::unique_ptr<DC>>,
    typename SegmentID<B,E>::SegmentIDHash
  >;

  typedef DepMap<0,0> IntactDeps_t;
  typedef DepMap<-1,0> RisingDeps_t;
  typedef DepMap<1,0> MayDangleDeps_t;
  typedef DepMap<0,-1> DanglingDeps_t;
  typedef DepMap<-1,-1> RisingDanglingDeps_t;
  typedef DepMap<1,-1> MayDangleDanglingDeps_t;
  typedef DepMap<0,1> MayRiseDeps_t;
  typedef DepMap<-1,1> MayRiseRisingDeps_t;
  typedef DepMap<1,1> MayRiseMayDangleDeps_t;

  std::unique_ptr<llvm::DC<LKMMAnnotateDeps> *> DCs;
};


//===----------------------------------------------------------------------===//
// The Hook Pass
//===----------------------------------------------------------------------===//

/// A wrapper around LKMMAnnotateDepsPass, that is able to be inserted into
/// the earliest hook point.
class LKMMAnnotateHook : public PassInfoMixin<LKMMAnnotateHook> {
public:

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {

    auto &Annotations = AM.getResult<LKMMAnnotateDepsPass>(M);
    return PreservedAnalyses::all();
  }
};
  
//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

class LKMMVerifyDepsPass : public PassInfoMixin<LKMMVerifyDepsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    
    auto &Annotations = AM.getResult<LKMMAnnotateDepsPass>(M);
    return PreservedAnalyses::all();
  };
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CUSTOMMEMDEP_H
