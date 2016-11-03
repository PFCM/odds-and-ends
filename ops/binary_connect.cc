#

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

/* For whatever reason, it wasn't figuring out that a lambda had the right
types. Therefore, we will define this functinon explicitly */
Status shapeTransformation(::tensorflow::shape_inference::InferenceContext* c){
  c->set_output(0, c->input(0));
  return Status::OK();
}

REGISTER_OP("BinaryConnect")
  .Attr("min_: float = -1.0")
  .Attr("max_: float =  1.0")
  .Input("activations: float")
  .Output("binarized: float")
  .SetShapeFn(shapeTransformation);

class BinaryConnectOp : public OpKernel {
public:
  explicit BinaryConnectOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_", &min_val_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_", &max_val_));
    // check that min < max
    OP_REQUIRES(context, min_val_ < max_val_,
                errors::InvalidArgument("Need min < max"));
  }

  void Compute(OpKernelContext* context) override {
    // get input
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // create output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0, input_tensor.shape(),
                                                    &output_tensor));
    auto output = output_tensor->flat<float>();
    const int N = input.size();

    // undoubtedly could be much more efficient
    // would love to avoid this branching
    for (int i = 0; i < N; i++) {
      auto val = input(i);
      output(i) = (val < 0.0f)? min_val_: max_val_;
    }
  }

private:
  float min_val_, max_val_;
};

REGISTER_KERNEL_BUILDER(Name("BinaryConnect").Device(DEVICE_CPU),
                        BinaryConnectOp)
REGISTER_KERNEL_BUILDER(Name("BinaryConnect").Device(DEVICE_GPU),
                        BinaryConnectOp)
