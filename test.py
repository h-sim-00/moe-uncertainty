from model.deepseekmoe.modeling_deepseek import DeepseekForCausalLM
from model.deepseekmoe.configuration_deepseek import DeepseekConfig

def main():
    model = DeepseekForCausalLM(DeepseekConfig())
    print(model)

if __name__ == "__main__":
    main()