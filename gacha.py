from models import ItemTable


def gacha(text: str) -> str:
    try:
        table = ItemTable.loads(text=text).optimize()
    except Exception:
        return (
            "❌ 失敗: 入力方法が正しくありません！\n"
            '"?gacha" コマンドで使い方を確認してください'
        )

    try:
        table.set_caches()
    except TimeoutError:
        return "❌ 失敗: 計算がタイムアウトしました！（組み合わせが多すぎます！）"
    except Exception as e:
        return f"❌ 失敗: 予期しないエラーが発生しました！\n```\n{e}\n```"

    result = ""
    pdf: list[float] | None = None

    try:
        pdf = table.calc_pdf()
    except TimeoutError:
        result += (
            "⚠️ 情報: 分布の計算がタイムアウトしました！（組み合わせが多すぎます！）\n"
        )
    except Exception as e:
        return f"❌ 失敗: 予期しないエラーが発生しました！\n```\n{e}\n```"

    properties = table.describe(pdf)
    result += "```\n"

    for key, value in properties.items():
        if key in {"平均", "標準偏差"}:
            result += f"{key}: {value:.2f}回\n"
        else:
            result += f"{key}: {value:.0f}回\n"

    result += "```"

    return result


if __name__ == "__main__":
    print(gacha("★1, 1/100"))
