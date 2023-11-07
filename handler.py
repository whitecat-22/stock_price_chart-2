try:
    #from notifiers import unzip_requirements
    import unzip_requirements
except ImportError:
    print("Import Error - unzip_requirements")
    pass
except Exception as e:
    print(e)
    pass

import csv
import datetime
import os
from os.path import join, dirname
import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dotenv import load_dotenv
#from notifiers import slack
#from notifiers import twitter
import tweepy
from decimal import Decimal, ROUND_HALF_UP
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
import time
import json
import logging

# settins for logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load env variants
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
load_dotenv(verbose=True)
stock_code = os.environ.get("STOCK_CODE")

is_today = "N"

# Get today"s date for getting the stock price and csv&image filename
today = datetime.date.today()

# tmp directory is present by default on Cloud Functions, so guard it
if not os.path.isdir("/tmp"):
    os.mkdir("/tmp")


"""
json: Format the data to be sent by the SLack API into JSON
requests: HTTP client
"""
# WebClient insantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
# ID of channel that you want to upload file to
token = os.environ.get("SLACK_BOT_TOKEN")
channel_id = os.environ.get("SLACK_CHANNEL_ID")

"""
json: Format the data to be sent by the Twitter API into JSON
requests: HTTP client
"""
# 各種twitterのKeyをセット CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_KEY_SECRET
CONSUMER_KEY = os.environ.get("CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

# tweepyの設定
# auth = tweepy.Client(BEARER_TOKEN)
auth = tweepy.OAuth1UserHandler(
    CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# auth = tweepy.OAuth2AppHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth)
client_t = tweepy.Client(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
)

class Slack():
    """
    Notification Class to configure the settings for the Slack API
    """
    def __init__(self, date, ohlcv):
        self.__date = date
        self.text = self.__format_text(ohlcv)

    @property
    def date(self):
        """
        Property of date to be displayed in Slack text
        :return: datetime
        """
        return self.__date

    def __format_text(self, ohlcv):
        """
        Create params data for sending Slack notification with API.
        :param dict[str, str, str, str, str, str] ohlcv:
        :type ohlcv: {
            "datetime": "2020-12-29",
            "open": "7620",
            "high": "8070",
			"low": "7610",
			"close": "8060",
			"volume": "823700"
        }
        :return: String
        """
        open_ = Decimal(str(ohlcv["open"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        high_ = Decimal(str(ohlcv["high"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        low_ = Decimal(str(ohlcv["low"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        close_ = Decimal(str(ohlcv["close"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)

        text = f"本日は{self.date.strftime('%Y年%m月%d日')}です。\n" \
               f"取得可能な最新日付の株価情報をお知らせします。 \n\n"\
               f"*銘柄*  {str(stock_code)}\n" \
               f"*日付*  {ohlcv['date']}\n" \
               f"*始値*  {str(open_)}\n" \
               f"*高値*  {str(high_)}\n" \
               f"*安値*  {str(low_)}\n" \
               f"*終値*  {str(close_)}\n" \
               f"*出来高*  {float(ohlcv['volume'])}"
        return text

    def post(self):
        """
        POST request to Slack file upload API
        API docs: https://slack.com/api/files.upload
        """
        # The name of the file you"re going to upload
        file = open(f"/tmp/{self.date}.jpg", "rb")
        title = f"{self.date}.jpg"
        # Call the files.upload method using the WebClient
        # Uploading files requires the `files:write` scope
        try:
            # client.files_upload(
            #     channels=channel_id,
            #     initial_comment=self.text,
            #     file=file,
            #     title=title
            # )
            client.files_upload_v2(
                channel=channel_id,
                initial_comment=self.text,
                file=file,
                title=title
            )
        except Exception as e:
            print(e)


class Twitter():
    """
    Notification Class to configure the settings for the Twitter API
    """
    def __init__(self, date, ohlcv):
        self.__date = date
        self.text = self.__format_text(ohlcv)

    @property
    def date(self):
        """
        Property of date to be displayed in Slack text
        :return: datetime
        """
        return self.__date

    def __format_text(self, ohlcv):
        """
        Create params data for sending Twitter notification with API.
        :param dict[str, str, str, str, str, str] ohlcv:
        :type ohlcv: {
            "datetime": "2020-12-29",
            "open": "7620",
            "high": "8070",
			"low": "7610",
			"close": "8060",
			"volume": "823700"
        }
        :return: String
        """
        open_ = Decimal(str(ohlcv["open"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        high_ = Decimal(str(ohlcv["high"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        low_ = Decimal(str(ohlcv["low"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        close_ = Decimal(str(ohlcv["close"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)

        text = f"本日は{self.date.strftime('%Y年%m月%d日')}です。\n" \
               f"取得可能な最新日付の株価情報をお知らせします。 \n\n"\
               f"銘柄  {str(stock_code)}\n" \
               f"日付  {ohlcv['date']}\n" \
               f"始値  {str(open_)}\n" \
               f"高値  {str(high_)}\n" \
               f"安値  {str(low_)}\n" \
               f"終値  {str(close_)}\n" \
               f"出来高  {float(ohlcv['volume'])}"
        return text

    def post(self):
        """
        POST request to Twitter API
        API docs: https://developer.twitter.com/en/docs/twitter-api/api-reference-index
        """
        # The name of the file you"re going to upload
        # file = open(f"/tmp/{self.date}.jpg", "rb")
        title = f"{self.date}.jpg"
        # Call the files.upload method using the WebClient
        # Uploading files requires the `files:write` scope
        try:
            file_names = ["/tmp/" + title, ]
            media_ids = []
            for filename in file_names:
                res = api.media_upload(filename=filename)
                media_ids.append(res.media_id)
                res2 = api.create_media_metadata(res.media_id, title)
            # tweet with multiple images
            client_t.create_tweet(text=self.text,
                              media_ids=media_ids)
        except Exception as e:
            print(e)


def generate_stock_chart_image(df, d_breaks):
    """
    Generate a six-month stock chart image with mplfinance
    """
    # dataframe = pd.read_csv(
    #     f"/tmp/{today}.csv", index_col=0, parse_dates=True)
    dataframe = df.copy()
    # The return value `datetime` from yahoofinance is sorted by asc, so change it to desc for plot
    dataframe = dataframe.sort_values("datetime")

    # figを定義
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.005, row_width=[0.1, 0.2, 0.2, 0.5])

    # ローソク足：Candlestick
    fig.add_trace(
        go.Candlestick(x=dataframe["datetime"], open=dataframe["open"], high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], name="株価"),
        row=1, col=1
    )

    # 一目均衡表
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["base_line"], name="基準線", mode="lines", line=dict(color="purple")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["conversion_line"], name="転換線", mode="lines", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["leading_span1"], name="先行スパン1", mode="lines", fill=None, line=dict(width=0, color="gray"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["leading_span2"], name="先行スパン2", mode="lines", fill='tonexty', line=dict(width=0, color="gray"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["lagging_span"], name="遅行線", mode="lines", line=dict(color="turquoise")), row=1, col=1)

    # SMA
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["SMA25"], name="SMA25", mode="lines", line=dict(color="magenta")), row=1, col=1)

    # ボリンジャーバンド
    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["2upper"], name="2σ", line=dict(width=1, color="pink")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["2lower"], line=dict(width=1, color="pink"), showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["3upper"], name="3σ", line=dict(width=1, color="skyblue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["3lower"], line=dict(width=1, color="skyblue"), showlegend=False),
        row=1, col=1
    )

    # 帯をつける(2upper-2lower)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["2upper"], mode="lines", fill=None, line=dict(width=0, color="pink"), showlegend=False), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["2lower"], mode="lines", fill='tonexty', line=dict(width=0, color="pink"), showlegend=False), row=1, col=1)

    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["3upper"], name="3σ", line=dict(width=1, color="blue")), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["3lower"], line=dict(width=1, color="blue"), showlegend=False), row=1, col=1)

    # 帯をつける(3upper-2upper, 2lower-3lower)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["3upper"], mode="lines", fill=None, line=dict(width=0, color="lightblue"), showlegend=False), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["2upper"], mode="lines", fill='tonexty', line=dict(width=0, color="lightblue"), showlegend=False), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["2lower"], mode="lines", fill=None, line=dict(width=0, color="lightblue"), showlegend=False), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["3lower"], mode="lines", fill='tonexty', line=dict(width=0, color="lightblue"), showlegend=False), row=1, col=1)


    # MACD
    # fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe["MACD"], mode="lines", showlegend=False), row=2, col=1)
    # fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe["Signal"], mode="lines", showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Bar(x=dataframe["datetime"], y=dataframe["MACD"], name="MACD", marker_color="gray"),
        row=2, col=1
    )

    # RSI
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe["RSI"], mode="lines", name="RSI", line=dict(color="blue")), row=3, col=1)

    # 出来高
    fig.add_trace(
        go.Bar(x=dataframe["datetime"], y=dataframe["volume"], name="出来高", marker_color="green"),
        row=4, col=1
    )

    # Layout
    fig.update_layout(
        title={
            "text": f"{stock_code}の日足チャート",
            "y":0.9,
            "x":0.5,
        },
        height=600,
    )

    # y軸名を定義
    fig.update_yaxes(title_text="株価", row=1, col=1, separatethousands=True)
    fig.update_yaxes(title_text="MACD", row=2, col=1, separatethousands=True)
    fig.update_yaxes(title_text="RSI", row=3, col=1, separatethousands=True)
    fig.update_yaxes(title_text="出来高", row=4, col=1, separatethousands=True)

    # 不要な日付を非表示にする
    fig.update_xaxes(
        rangebreaks=[dict(values=d_breaks)],
        tickformat='%Y/%m/%d',
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    #fig.show()
    fig.write_image(f"/tmp/{today}.jpg")

    return dataframe


def generate_csv_from_dataframe():
    global is_today
    # is_today = "Y"  # for test run
    """
    Generate dataframe of OHLCV with date by yahoo_finance_api2
    """

    def macd(df):
        FastEMA_period = 12  # 短期EMAの期間
        SlowEMA_period = 26  # 長期EMAの期間
        SignalSMA_period = 9  # SMAを取る期間
        df["MACD"] = df["close"].ewm(span=FastEMA_period).mean() - df["close"].ewm(span=SlowEMA_period).mean()
        df["Signal"] = df["MACD"].rolling(SignalSMA_period).mean()
        return df

    def rsi(df):
        # 前日との差分を計算
        df_diff = df["close"].diff(1)

        # 計算用のDataFrameを定義
        df_up, df_down = df_diff.copy(), df_diff.copy()

        # df_upはマイナス値を0に変換
        # df_downはプラス値を0に変換して正負反転
        df_up[df_up < 0] = 0
        df_down[df_down > 0] = 0
        df_down = df_down * -1

        # 期間14でそれぞれの平均を算出
        df_up_sma14 = df_up.rolling(window=14, center=False).mean()
        df_down_sma14 = df_down.rolling(window=14, center=False).mean()

        # RSIを算出
        df["RSI"] = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))

        return df

    # yahoo_finance_api2で過去2年分の株価を取得する
    my_share = share.Share(stock_code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(
            share.PERIOD_TYPE_YEAR, 2,
            share.FREQUENCY_TYPE_DAY, 1)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    df = pd.DataFrame(symbol_data)
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    # APIで取得したデータを一旦CSVファイルにする
    df1 = df.copy()
    df1 = df1.sort_values(by="datetime", ascending=False)
    df1.to_csv(f"/tmp/{today}.csv")

    wk_date = df1.iloc[0]["datetime"].strftime("%Y-%m-%d")
    wk_today = today.strftime("%Y-%m-%d")
    if wk_date == wk_today:
        is_today = "Y"

    additional_dates = pd.date_range(
        start=df["datetime"].max()+datetime.timedelta(days=1),
        end=df["datetime"].max()+datetime.timedelta(days=25),
    )

    df = pd.concat([
        df,
        pd.DataFrame(additional_dates, columns=["datetime"])
    ], ignore_index=True)


    # 基準線
    high26 = df["high"].rolling(window=26).max()
    low26 = df["low"].rolling(window=26).min()
    df["base_line"] = (high26 + low26) / 2

    # 転換線
    high9 = df["high"].rolling(window=9).max()
    low9 = df["low"].rolling(window=9).min()
    df["conversion_line"] = (high9 + low9) / 2

    # 先行スパン1
    leading_span1 = (df["base_line"] + df["conversion_line"]) / 2
    df["leading_span1"] = leading_span1.shift(25)

    # 先行スパン2
    high52 = df["high"].rolling(window=52).max()
    low52 = df["low"].rolling(window=52).min()
    leading_span2 = (high52 + low52) / 2
    df["leading_span2"] = leading_span2.shift(25)

    # 遅行スパン
    df["lagging_span"] = df["close"].shift(-25)

    # 25日移動平均線
    df["SMA25"] = df["close"].rolling(window=25).mean()

    # 標準偏差
    df["std"] = df["close"].rolling(window=25).std()

    # ボリンジャーバンド
    df["2upper"] = df["SMA25"] + (2 * df["std"])
    df["2lower"] = df["SMA25"] - (2 * df["std"])
    df["3upper"] = df["SMA25"] + (3 * df["std"])
    df["3lower"] = df["SMA25"] - (3 * df["std"])

    # MACDを計算する
    df = macd(df)

    # RSIを算出
    df = rsi(df)

    # 非表示にする日付をリストアップ
    d_all = pd.date_range(start=df['datetime'].iloc[0],end=df['datetime'].iloc[-1])
    d_obs = [d.strftime("%Y-%m-%d") for d in df['datetime']]
    d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if d not in d_obs]

    return [df, d_breaks, is_today]

def lambdahandler(event, context):
    global is_today
    """
    lambda_handler
    """
    logging.info(json.dumps(event))

    print("event: {}".format(event))
    print("context: {}".format(context))
    """
    The main function that will be executed when this Python file is executed
    """
    result = generate_csv_from_dataframe()
    df = result[0]
    d_breaks = result[1]
    is_today = result[2]

    if is_today == "Y":

        pio.kaleido.scope.chromium_args += ("--single-process",)
        cold = True
        while cold:
            try:
                generate_stock_chart_image(df, d_breaks)
                cold = False
            except Exception as ex:
                print(f"WARMUP EXCEPTION {ex}")

            if cold:
                time.sleep(10)

            generate_stock_chart_image(df, d_breaks)

            with open(f"/tmp/{today}.csv", "r", encoding="utf-8") as file:
                # Skip header row
                reader = csv.reader(file)
                header = next(reader)
                for i, row in enumerate(csv.DictReader(file, header)):
                    # Send only the most recent data to Slack notification
                    if i == 0:
                        Slack(today, row).post()
                        Twitter(today, row).post()

    return {
        "statusCode": 200,
        "body": "ok"
    }


if __name__ == "__main__":
    print(lambdahandler(event=None, context=None))
