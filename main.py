from dotenv import load_dotenv
import os
import discord

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からトークンを取得する
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# すべてのイベントに対して反応するIntentsオブジェクトを作成
intents = discord.Intents.default()
intents.message_content = True # メッセージの内容を受け取るための権限を付与

# Discordのクライアントを作成
client = discord.Client(intents=intents)

@client.event
async def on_ready():
  print(f'{client.user}としてログインしました')

@client.event
async def on_message(message):
  # Bot自身のメッセージには反応しないようにする
  if message.author.bot:
    return
  
  # 'Botさん、こんにちは！'というメッセージに反応する
  if message.content == 'Botさん、こんにちは！':
    # メッセージ送信者の表示名を取得
    user_name = message.author.display_name
    # 応答するメッセージを作成
    response = f'こんにちは、{user_name}さん！'
    await message.channel.send(response)

client.run(TOKEN)