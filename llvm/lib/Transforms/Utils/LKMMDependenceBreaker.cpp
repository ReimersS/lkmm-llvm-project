#include "llvm/Transforms/Utils/LKMMDependenceBreaker.h"

namespace llvm {


  PreservedAnalyses LKMMBreakingPass::run(Module &M, ModuleAnalysisManager &AM) {

    //auto &LKMMAnnotateDeps = AM


    return PreservedAnalyses::all(); // This is of course not true, but that is the point
  }
} // namespace llvm
