import argparse
import glob
import sys
from typing import Dict

import yaml

from deid import main


class TestMain:
    def __init__(
            self,
            key: str,
            deid_cls_name: str,
            deid_cls_args: Dict,
            txt_folder: str
        ):
        self.key = key
        main.register(key, deid_cls_name, deid_cls_args)
        self.txt_path_list = sorted(glob.glob(f"{txt_folder}/*.txt"))
        assert len(self.txt_path_list) > 0, f"No txt found in {txt_folder}."

    def test_run(self):
        pass_list = []
        for txt_path in self.txt_path_list:
            print(f"========== txt_path: {txt_path} ==========")
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            out = main.run(self.key, raw_text)
            pass_list.append(not out.startswith("[Deid_failed]"))
        main.unregister(key=self.key)
        assert all(pass_list), f"Some deid failed. pass_list: {pass_list}"
        print("All passed :D")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_folder",
        "-e",
        type=str, 
        default="/app/exps/main/example"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"{args.exp_folder}/cfg.yaml", "r", encoding="utf-8"))['cfg']

    test_cls = getattr(sys.modules[__name__], cfg['cls_name'])
    test_obj = test_cls(**cfg['args'])
    test_obj.test_run()
