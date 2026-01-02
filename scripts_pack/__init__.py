"""
批量攻击脚本包

Usage 示例请参考同目录下的 batch_attack.py

先用dry-run测试一下
python -m scripts_pack.batch_attack --model-prefix pythia-70m --attacks loss --dry-run

所有的策略和所有的攻击
python -m scripts_pack.batch_attack `
  --model-prefix pythia-70m `
  --strategies full lora head pretrained`
  --attacks loss ratio mink `
  --dry-run
"""


