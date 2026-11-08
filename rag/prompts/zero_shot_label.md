You help an analyst who watches US equities. Below is a post that Donald Trump published on Truth Social during US market hours.

Task: estimate the probability that the S&P 500 makes an abnormally large move over the next two hourly bars after the post. A move is abnormal when its absolute log return is more than 1.5 standard deviations of the same time-slot return over the previous 20 trading days. Roughly 14 posts out of 100 are followed by an abnormal move, so most posts are not.

Post time (New York): {{time_et}}
Post text:
"""
{{text}}
"""

Answer with one JSON object and nothing else: {"probability": <number between 0 and 1>}
