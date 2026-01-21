import requests

url = "http://localhost:8000/news/"
true_data = {
    "title": "Mexico to review need for tax changes after U.S. reform-document MEXICO CITY (Reuters)",
    "text": "Mexico’s finance ministry will predict whether to make fiscal changes in response to the U.S. tax reform, according to a document seen by Reuters on Friday. In the document, the ministry said Mexico would not make changes that left it with a higher public sector deficit. “Nevertheless, there will be an assessment of whether modifications should be made to Mexico’s fiscal framework,” the document said.",
}
fake_data = {
    "title": "The Netherlands Just TROLLED Trump In His Own Words And It’s Hilariously BRILLIANT (VIDEO)",
    "text": " While Donald Trump likes to go around saying he s only going to put America first, the rest of the world has its own opinion about that.One of the nations speaking out against Trump just absolutely and hilariously satirized him using his own words   or rather, words he s notoriously been known to use with his figures of speech. That nation is The Netherlands.The satire show Zondag Met Lubach, much like our own The Daily Show, put together a fake tourism commercial using a voice that sounded just like Trump s, and using language that he would use, including some policy proposals.Needless to say, they re not huge fans of Trump whatsoever.Dutch comedian Arjen Lubach says, as he introduces the pretend advertisement: He had a clear message to the rest of the world:  I will screw you over big time.  Well okay, he used slightly different words:  From this day forward. It s going to be only America first .And we realize it s better for us to get along [so] we decided to introduce our tiny country to him in a way that will probably appeal to him the most. From there, it was absolutely comic genius.Watch the hilarious video here:Featured image via screen capture from embedded video",
}


# data = {"title": "foo", "text": "bar"}
# response = requests.post(url, data=form)
# print(response.text)

response = requests.post(url, json=true_data)
print(response.text)
response = requests.post(url, json=fake_data)
print(response.text)
