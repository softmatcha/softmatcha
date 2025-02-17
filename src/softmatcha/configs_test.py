from softmatcha.configs import OutputArguments, get_argparser


class TestArgumentParser:
    def test_config_load(self):
        cfg_dict = {
            "common": {
                "backend": "fasttext",
                "model": "fasttext-en-vectors",
            },
            "output": {"line_number": True, "log": "-"},
        }
        cmd_args = [
            "--backend",
            "fasttext",
            "--model",
            "fasttext-en-vectors",
            "--line_number",
            "--log",
        ]

        parser = get_argparser(args=cmd_args)
        parser.add_arguments(OutputArguments, "output")
        args = parser.parse_args(args=cmd_args)
        for key, value in cfg_dict["common"].items():
            assert getattr(args.common, key) == value
        for key, value in cfg_dict["output"].items():
            assert getattr(args.output, key) == value
