try:
    # from notifiers import unzip_requirements
    import unzip_requirements
except ImportError:
    print("Import Error - unzip_requirements")
except Exception as e:
    print(e)

import csv
import datetime
import os
from os.path import join, dirname
import time
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dotenv import load_dotenv
import dateutil.relativedelta
# from notifiers import slack
# from notifiers import twitter
import tweepy
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient

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
two_year_ago = today - dateutil.relativedelta.relativedelta(years=2)

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
            "Open": "7620",
            "High": "8070",
            "Low": "7610",
            "Close": "8060",
            "Adj Close": "8060",
            "Volume": "823700"
        }
        :return: String
        """
        open_ = Decimal(str(ohlcv["Open"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        high_ = Decimal(str(ohlcv["High"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        low_ = Decimal(str(ohlcv["Low"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        close_ = Decimal(str(ohlcv["Close"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        day_before_ = Decimal(str(ohlcv["day_before"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        if day_before_ > 0:
            day_before_str_ = f"（前日比  ＋{str(abs(day_before_))}円）"
        elif day_before_ == 0:
            day_before_str_ = f"（前日比  ±{str(abs(day_before_))}円）"
        else:  # if day_before_ < 0:
            day_before_str_ = f"（前日比  ▲{str(abs(day_before_))}円）"

        text = f"本日は{self.date.strftime('%Y年%m月%d日')}です。\n" \
               f"取得可能な最新日付の株価情報をお知らせします。 \n\n"\
               f"*銘柄*  {str(stock_code)}\n" \
               f"*日付*  {ohlcv['datetime']}\n" \
               f"*始値*  {str(open_)}円\n" \
               f"*高値*  {str(high_)}円\n" \
               f"*安値*  {str(low_)}円\n" \
               f"*終値*  {str(close_)}円  {day_before_str_}\n" \
               f"*出来高*  {float(ohlcv['Volume'])}"
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
            "Open": "7620",
            "High": "8070",
            "Low": "7610",
            "Close": "8060",
            "Adj Close": "8060",
            "Volume": "823700"
        }
        :return: String
        """
        open_ = Decimal(str(ohlcv["Open"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        high_ = Decimal(str(ohlcv["High"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        low_ = Decimal(str(ohlcv["Low"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        close_ = Decimal(str(ohlcv["Close"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        day_before_ = Decimal(str(ohlcv["day_before"])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        if day_before_ > 0:
            day_before_str_ = f"（前日比  ＋{str(abs(day_before_))}円）"
        elif day_before_ == 0:
            day_before_str_ = f"（前日比  ±{str(abs(day_before_))}円）"
        else:  # if day_before_ < 0:
            day_before_str_ = f"（前日比  ▲{str(abs(day_before_))}円）"

        text = f"本日は{self.date.strftime('%Y年%m月%d日')}です。\n" \
               f"取得可能な最新日付の株価情報をお知らせします。 \n\n"\
               f"銘柄  {str(stock_code)}\n" \
               f"日付  {ohlcv['datetime']}\n" \
               f"始値  {str(open_)}円\n" \
               f"高値  {str(high_)}円\n" \
               f"安値  {str(low_)}円\n" \
               f"終値  {str(close_)}円  {day_before_str_}\n" \
               f"出来高  {float(ohlcv['Volume'])}"
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
                api.create_media_metadata(res.media_id, title)
            # tweet with multiple images
            client_t.create_tweet(
                text=self.text,
                media_ids=media_ids,
            )
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
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.005, row_width=[0.1, 0.1, 0.2, 0.2, 0.5])

    # ローソク足：Candlestick
    fig.add_trace(
        go.Candlestick(x=dataframe["datetime"], open=dataframe["Open"], high=dataframe["High"], low=dataframe["Low"], close=dataframe["Close"], name="Prices"),
        row=1, col=1
    )

    # 一目均衡表
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["base_line"], name="BaseLine", mode="lines", line=dict(color="purple")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["conversion_line"], name="Conv.Line", mode="lines", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["leading_span1"], name="AdvanceSpan1", mode="lines", fill=None, line=dict(width=0, color="gray"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["leading_span2"], name="AdvanceSpan2", mode="lines", fill='tonexty', line=dict(width=0, color="gray"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["lagging_span"], name="LaggingSpan", mode="lines", line=dict(color="turquoise")), row=1, col=1)

    # SMA
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["SMA25"], name="SMA25", mode="lines", line=dict(color="magenta")), row=1, col=1)
    # fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["SMA50"], name="SMA50", mode="lines", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe["datetime"], y=dataframe["SMA200"], name="SMA200", mode="lines", line=dict(color="green")), row=1, col=1)

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

    # 買われすぎ、売られすぎサイン
    # 25, s25 = dataframe["SMA25_dr"].mean(), dataframe["SMA25_dr"].std()
    # m50, s50 = dataframe["SMA50_dr"].mean(), dataframe["SMA50_dr"].std()
    # m00, s200 = dataframe["SMA200_dr"].mean(), dataframe["SMA200_dr"].std()
    # 買われすぎ
    # fig.add_trace(go.Scatter(x=dataframe[dataframe["SMA25_dr"]>(m25+(2*s25))]["datetime"], y=dataframe[dataframe["SMA25_dr"]>(m25+(2*s25))]["Close"]*1.02, name="買われすぎ", mode="markers", marker_symbol="triangle-down", marker_size=5, marker_color="black"), row=1, col=1)
    # 売られすぎ
    # fig.add_trace(go.Scatter(x=dataframe[dataframe["SMA25_dr"]<(m25-(2*s25))]["datetime"], y=dataframe[dataframe["SMA25_dr"]<(m25-(2*s25))]["Close"]*0.98, name="売られすぎ", mode="markers", marker_symbol="triangle-up", marker_size=5, marker_color="black"), row=1, col=1)


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
        go.Bar(x=dataframe["datetime"], y=dataframe["Volume"], name="Volume", marker_color="green"),
        row=4, col=1
    )

    # 乖離率
    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["SMA25_dr"], name="SMA25_%", mode="lines", line=dict(color="magenta")),
        row=5, col=1
    )
    # fig.add_trace(
    #     go.Scatter(x=dataframe["datetime"], y=dataframe["SMA50_乖離率"], name="SMA50_%", mode="lines", line=dict(color="blue")),
    #     row=5, col=1
    # )
    fig.add_trace(
        go.Scatter(x=dataframe["datetime"], y=dataframe["SMA200_dr"], name="SMA200_%", mode="lines", line=dict(color="green")),
        row=5, col=1
    )

    # Layout
    fig.update_layout(
        plot_bgcolor="white",
        title={
            "text": f"Daily chart of {stock_code}",
            "y": 0.9,
            "x": 0.5,
        },
        width=2000,
        height=1000,
    )

    # y軸名を定義
    fig.update_yaxes(title_text="Prices", row=1, col=1, separatethousands=True, showline=True, linewidth=1, linecolor="lightgrey", color="grey")
    fig.update_yaxes(title_text="MACD", row=2, col=1, separatethousands=True, showline=True, linewidth=1, linecolor="lightgrey", color="grey")
    fig.update_yaxes(title_text="RSI", row=3, col=1, separatethousands=True, showline=True, linewidth=1, linecolor="lightgrey", color="grey")
    fig.update_yaxes(title_text="Vol", row=4, col=1, separatethousands=True, showline=True, linewidth=1, linecolor="lightgrey", color="grey")
    fig.update_yaxes(title_text="%", row=5, col=1, separatethousands=True, showline=True, linewidth=1, linecolor="lightgrey", color="grey")

    # 不要な日付を非表示にする
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="lightgrey", color="grey",
        rangebreaks=[dict(values=d_breaks)],
        tickformat='%Y/%m/%d',
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    # fig.show()
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
        df["MACD"] = df["Close"].ewm(span=FastEMA_period).mean() - df["Close"].ewm(span=SlowEMA_period).mean()
        df["Signal"] = df["MACD"].rolling(SignalSMA_period).mean()
        return df

    def rsi(df):
        # 前日との差分を計算
        df_diff = df["Close"].diff(1)

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

    # yfinanceで過去5年分の株価を取得する
    df = yf.download(stock_code, period= "5y", interval = "1d")
    df["day_before"] = df["Close"].diff(1)
    df["datetime"] = pd.to_datetime(df.index, unit="ms")

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
    high26 = df["High"].rolling(window=26).max()
    low26 = df["Low"].rolling(window=26).min()
    df["base_line"] = (high26 + low26) / 2

    # 転換線
    high9 = df["High"].rolling(window=9).max()
    low9 = df["Low"].rolling(window=9).min()
    df["conversion_line"] = (high9 + low9) / 2

    # 先行スパン1
    leading_span1 = (df["base_line"] + df["conversion_line"]) / 2
    df["leading_span1"] = leading_span1.shift(25)

    # 先行スパン2
    high52 = df["High"].rolling(window=52).max()
    low52 = df["Low"].rolling(window=52).min()
    leading_span2 = (high52 + low52) / 2
    df["leading_span2"] = leading_span2.shift(25)

    # 遅行スパン
    df["lagging_span"] = df["Close"].shift(-25)

    # 25日移動平均線
    df["SMA25"] = df["Close"].rolling(window=25).mean()
    # 50日移動平均線
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    # 200日移動平均線
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # 移動平均線乖離率 Deviation Ratio
    df["SMA25_dr"] = ((df["Close"] / df["SMA25"]) - 1) * 100
    df["SMA50_dr"] = ((df["Close"] / df["SMA50"]) - 1) * 100
    df["SMA200_dr"] = ((df["Close"] / df["SMA200"]) - 1) * 100

    # 標準偏差
    df["std"] = df["Close"].rolling(window=25).std()

    # ボリンジャーバンド
    df["2upper"] = df["SMA25"] + (2 * df["std"])
    df["2lower"] = df["SMA25"] - (2 * df["std"])
    df["3upper"] = df["SMA25"] + (3 * df["std"])
    df["3lower"] = df["SMA25"] - (3 * df["std"])

    # MACDを計算する
    df = macd(df)

    # RSIを算出
    df = rsi(df)

    df.index = pd.to_datetime(df["datetime"], format="%Y-%m-%d").values
    df = df[two_year_ago : today]

    # 非表示にする日付をリストアップ
    d_all = pd.date_range(start=df["datetime"].iloc[0], end=df["datetime"].iloc[-1])
    d_obs = [d.strftime("%Y-%m-%d") for d in df["datetime"]]
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
