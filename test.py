from model.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from model.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

def main():
    model = Qwen2MoeForCausalLM(Qwen2MoeConfig())
    print(model)

if __name__ == "__main__":
    main()