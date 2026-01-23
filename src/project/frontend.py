import gradio as gr
import requests

API_URL = "http://localhost:8000"


def predict(title: str, text: str) -> str:
    """Send prediction request to FastAPI backend."""
    if not title.strip() and not text.strip():
        return "Please enter a title or text"

    try:
        response = requests.post(
            f"{API_URL}/news/",
            json={"title": title, "text": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.text.strip('"')
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to API. Make sure it's running on localhost:8000"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def get_model_info() -> str:
    """Get current model info from API."""
    try:
        response = requests.get(f"{API_URL}/model/", timeout=10)
        return f"Model: {response.text}"
    except requests.exceptions.RequestException:
        return "Model info unavailable"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Title", placeholder="Enter news headline..."),
        gr.Textbox(label="Text", placeholder="Enter news article text...", lines=8),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fake News Detector",
    description="Enter a news article to classify it as real or fake news. API must be running on localhost:8000.\n\n**Examples below:** First 5 are FAKE news, last 5 are REAL news (from Reuters).",
    examples=[
        # FAKE NEWS examples (label=0)
        [
            "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
            "Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media.",
        ],
        [
            "Drunk Bragging Trump Staffer Started Russian Collusion Investigation",
            "House Intelligence Committee Chairman Devin Nunes is going to have a bad day. Former Trump campaign adviser George Papadopoulos was drunk in a wine bar when he revealed knowledge of Russian opposition research.",
        ],
        [
            "Sheriff David Clarke Becomes An Internet Joke For Threatening To Poke People In The Eye",
            "On Friday, it was revealed that former Milwaukee Sheriff David Clarke has an email scandal of his own. Clarke is calling it fake news even though copies of the search warrant are on the Internet.",
        ],
        [
            "Trump Is So Obsessed He Even Has Obama's Name Coded Into His Website",
            "On Christmas day, Donald Trump announced that he would be back to work the following day, but he is golfing for the fourth day in a row.",
        ],
        [
            "Pope Francis Just Called Out Donald Trump During His Christmas Speech",
            "Pope Francis used his annual Christmas Day message to rebuke Donald Trump without even mentioning his name.",
        ],
        # REAL NEWS examples (label=1)
        [
            "As U.S. budget fight looms, Republicans flip their fiscal script",
            "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a fiscal conservative on Sunday.",
        ],
        [
            "U.S. military to accept transgender recruits on Monday: Pentagon",
            "WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said on Friday.",
        ],
        [
            "Senior U.S. Republican senator: Let Mr. Mueller do his job",
            "WASHINGTON (Reuters) - The special counsel investigation of links between Russia and President Trump's 2016 election campaign should continue without interference in 2018, a prominent Republican senator said on Sunday.",
        ],
        [
            "FBI Russia probe helped by Australian diplomat tip-off: NYT",
            "WASHINGTON (Reuters) - Trump campaign adviser George Papadopoulos told an Australian diplomat in May 2016 that Russia had political dirt on Democratic presidential candidate Hillary Clinton, the New York Times reported.",
        ],
        [
            "Trump wants Postal Service to charge much more for Amazon shipments",
            "SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on the U.S. Postal Service on Friday to charge much more to ship packages for Amazon, picking another fight with an online retail giant.",
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()
