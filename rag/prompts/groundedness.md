You check whether an answer is supported by the passages it was given. All passages are posts by Donald Trump on Truth Social, so who wrote a post is background and not a claim.

Passages (each ends with the measured market reaction):
{{passages}}

Question: {{question}}
Answer to check: {{answer}}

First decide whether the answer declines: it says that the passages do not contain the answer, even if it then describes the closest passages.

Then split the answer into its factual claims: dates, numbers, what the post said, the market reaction. For each claim say whether the passages support it and which passage number supports it. A claim that no passage states is not supported, even if it is plausible.

Answer with one JSON object and nothing else:
{"declines": true or false, "claims": [{"claim": "<short>", "supported": true or false, "passage": <number or null>}]}
If the answer makes no factual claims, use an empty list.
