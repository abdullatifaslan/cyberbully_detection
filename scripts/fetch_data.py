from telethon.sync import TelegramClient
from config.config import API_ID, API_HASH, CHANNEL_NAME
import pandas as pd


def fetch_telegram_messages(limit=100):
    with TelegramClient('anon', API_ID, API_HASH) as client:
        channel = client.get_entity(CHANNEL_NAME)
        messages = client.iter_messages(channel, limit=limit)

        data = []
        for message in messages:
            if message.message:
                data.append(message.message)

        df = pd.DataFrame(data, columns=['Text'])
        return df


if __name__ == "__main__":
    df = fetch_telegram_messages(limit=100)
    print(df)
