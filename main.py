# 说明：
# 本文件是程序的主入口，负责串联配置解析、文献扫描快照保存、向量库加载和命令行问答。
# 当前版本保持与原始单文件脚本相近的运行方式，但内部逻辑已经拆分到独立模块中。

from src.config import parse_args, resolve_config, print_config_summary
from src.indexer import get_vectorstore, create_chat_chain, answer_once, interactive_chat
from src.bootstrap import prepare_manifest_snapshot


def main() -> None:
    """
    程序主流程入口。
    它会依次完成配置解析、目录快照更新、向量库准备和问答模式选择。
    """
    args = parse_args()
    config = resolve_config(args)
    print_config_summary(config)

    prepare_manifest_snapshot(config)

    vectorstore = get_vectorstore(config)
    rag_chain = create_chat_chain(config, vectorstore)

    if config.question:
        answer_once(rag_chain, config.question)
    else:
        interactive_chat(rag_chain)


if __name__ == "__main__":
    main()