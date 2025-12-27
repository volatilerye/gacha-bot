from timeout_decorator import TimeoutError

from models import (
    DuplicatesValidationError,
    ItemTable,
    ProbabilityValidationError,
    RequiredValidationError,
)


def gacha(text: str) -> str:
    """Simulate gacha pulls based on the provided item table text.

    Args:
        text (str): The input text defining items, their probabilities, and required counts.

    Returns:
        str: The result of the gacha simulation or an error message.
    """
    try:
        table = ItemTable.loads(text=text).optimize()
    except ProbabilityValidationError as e:
        return (
            f"❌ **失敗**: 確率（または確率の整数比）の指定が誤っています\n```\n{e}\n```\n"
            "次のいずれかで指定してください\n"
            "- 全てのアイテムの確率を0より大きく1以下の値で指定する\n"
            "- 全てのアイテムの確率を正の整数による比で指定する\n"
        )
    except RequiredValidationError as e:
        return (
            f"❌ **失敗**: 必要数の指定が誤っています\n```\n{e}\n```\n"
            "必要数は非負整数で指定してください（省略した場合はデフォルトで1になります）"
        )
    except DuplicatesValidationError as e:
        return (
            f"❌ **失敗**: 重複数の指定が誤っています\n```\n{e}\n```\n"
            "必要数は非負整数で指定してください（省略した場合はデフォルトで1になります）"
        )
    except Exception as e:
        return (
            "❌ **失敗**: コマンドの指定が誤っています"
            '"?gacha" コマンドで使い方を確認してください'
        )

    # set caches (and calc average and standard deviation)
    try:
        table = table.set_mat_caches()
    except TimeoutError:
        return "⏱️ **失敗**: 確率遷移行列の計算がタイムアウトしました（組み合わせが多過ぎます！）"
    except Exception as e:
        return f"❌ **失敗**: 予期しないエラーが発生しました！\n```\n{e}\n```"

    # set averages
    try:
        table = table.set_average()
    except TimeoutError:
        return "⏱️ **失敗**: 平均回数の計算がタイムアウトしました（組み合わせが多過ぎます！）"
    except Exception as e:
        return f"❌ **失敗**: 予期しないエラーが発生しました！\n```\n{e}\n```"

    # calculate probability distribution
    result = ""
    pdf: list[float] | None = None

    if table.cache_ave >= 0:
        try:
            table = table.set_std()
        except TimeoutError:
            result += "⏱️ **中断**: 標準偏差の計算がタイムアウトしました（組み合わせが多過ぎます！）\n"
        except Exception as e:
            result += f"❌ **失敗**: 予期しないエラーが発生しました！\n```\n{e}\n```\n"

    if table.cache_std >= 0:
        try:
            pdf = table.calc_pdf()
        except TimeoutError:
            result += "⏱️ **中断**: 分布の計算がタイムアウトしました（アイテムの出現確率が低過ぎるか組み合わせが多過ぎます！）\n"
        except Exception as e:
            return f"❌ **失敗**: 予期しないエラーが発生しました！\n```\n{e}\n```\n"

    properties = table.describe(pdf)
    if len(properties) == 0:
        return result

    result += "```\n"

    for key, value in properties.items():
        if key in {"平均", "標準偏差"}:
            result += f"{key}: {value:.2f}回\n"
        else:
            result += f"{key}: {value:.0f}回\n"

    result += "```"

    return result
