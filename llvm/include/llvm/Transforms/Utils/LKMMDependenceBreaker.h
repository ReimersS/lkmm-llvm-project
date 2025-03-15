#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEBRAKING_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEBRAKING_H

#include "llvm/Transforms/Utils/LKMMDependenceAnalysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LKMMBreakingPass : public PassInfoMixin<LKMMBreakingPass> {
public:

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEBRAKING_H
