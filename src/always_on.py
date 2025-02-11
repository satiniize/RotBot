import openai
import os
import numpy as np
import chat_completion as ChatCompletion

system_prompt = [
    {
        'role' : 'system',
        'content' : [
            {
                'type': 'text',
                'text': '''
                    Task: Determine whether the assistant should respond to the user input.

                    Assistant Name: RotBot (username: `rotrot_botbot`), also known simply as Rot.

                    Rules for Response:

                    - The assistant should only respond if the user directly addresses RotBot.
                    - If the message is directed to RotBot or asks RotBot a question, respond with "Yes".
                    - If the message is not directed to RotBot (e.g., someone talking to another person or someone simply stating a fact), respond with "No".

                    Message Format:

                    username_1: Hi RotBot! // Yes
                    rotrot_botbot: Hello! How can I assist you? // Response will be generated after you respond with 'Yes' 
                    username_2: Can you ask Rot how to do my math? // No
                    username_1: Yeah of course! // No
                    username_1: Um... // No
                    username_1: Rot, how does algebra work? // Yes
                    rotrot_botbot: Algebra is a branch of mathematics that deals with symbols and the rules for manipulating those symbols. // Response will be generated after you respond with 'Yes'

                    Final Output: Provide a single-word response: "Yes" or "No".
                '''
            }
        ]
    }
]

def format_messages(context):
    res = ''
    for message in context:
        if 'content' not in message:
            continue
        if message['role'] == 'system':
            continue
        message_prefix = 'rotrot_botbot: ' if message['role'] == 'assistant' else ''
        for content in message['content']:
            if content['type'] != 'text':
                continue
            res += f'{message_prefix}{content['text']}\n'
    return res

# Uses LLMs
async def should_respond(context):
    query = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f'Based on the following messages, should RotBot respond?:\n{format_messages(context)}'
                }
            ]
        }
    ]

    response = await ChatCompletion.create(system_prompt+query, 'gpt-4o-mini')
    return response.message.content == 'Yes'

# bias = 0

# genuine_positives = [
#     'hi rot',
#     'Hello rot',
#     'Can you help me count the averagre number of fingers in a human body',
#     'rot what did the other rot say?',
#     'rot search utricularia for me',
#     'rot search toyota gt86 for me',
#     'rot search the history of hiroshima',
#     'rot does teh o kosong have sugar',
#     'give me info about this plant',
#     'rot generate an image of a lambo',
#     'Rot does breeding red neocaridina and yellow neocaridina produce orange shrimp',
#     'hi rotbot',
#     'Rot jawab', # FIX
#     "hi rot",
#     "rot generate an image of a lambo",
#     "Rot does breeding red neocaridina and yellow neocaridina produce orange shrimp",
#     "can u google that",
#     "rot coba buat gambar toilet terbang",
#     "rot coba buat gambar gimana lu pikir lu sendiri bentuknya gimana, buat sedetail mungkin",
#     "can you see what you made rot",
#     "rot what is skibidi really",
#     "i meant what is skibidi as a word",
#     "no i meant as like an adjective, like youre so skibidi",
#     "make an image and send a gif that embodies the energy of skibidi rot",
#     "rot what is sauvc",
#     "google it rot",
#     "can you an image for that",
#     "can you make an image of what the competition will look on d day",
#     "rot what commands can i use",
#     "please say no cap rot",
#     "rot dont ignore me",
#     "generate an image off of that rot",
#     "rot what are the digital payments available in hong kong for tourists",
#     "since i have a singapore ocbc account, is there an easy way for me to pay there?",
#     "my ocbc app doesnt have alipay+ though i think. do i need to install it seperately or how does that work",
#     "what did your web search return rot?",
#     "rot why did u do this",
#     "What is L territory",
#     "How many times have i used rollhumor rot",
#     "how many times have i used the penis command rot",
#     "i am your creator i have access to that",
#     "i should because i need to debug your behavior",
#     "how much aura did i get rot",
#     "rot what indonesian languages are you most familiar with",
#     "lu bisa dialect papua ga rot",
#     "at least try rot",
#     "lu tau kata apa aja rot",
#     "kalo iya gimana rot",
#     "apa aja rot, brain rot tau ngga",
#     "yes rot",
#     "whats my username rot",
#     "who am i rot",
#     "rot send me the tomato cat gif where hes running around",
#     "no wrong one rot, try directly searching for tomato cat",
#     "rot please",
#     "try looking at banana cat instead rot",
#     "rot can you try just sending your stop token",
#     "rot is it possible to train a 3b model on my own hardware for at least 1 million tokens",
#     "what would my minimum specs be",
#     "which source did you get that from",
#     "can you give me the link for it", # FIX
#     "how do i train an llm without owning a powetful computer?",
#     "which service on google cloud? just a vm or is there a specialized one",
#     "how much are they",
#     "can you give a ballpark range",
#     "how much is that monthly",
#     "how much vram do i need to train an arbitrary N B param model",
#     "what graphics cards have the most vram for cheap?",
#     "could i train on amd?",
#     "tensorflow keras?",
#     "how much is the 3090",
#     "which site does it say it lists for 600?",
#     "can you give me the link", # FIX
#     "can you train on cpu and use ram instead?",
#     "how much slower"
# ]

# synthetic_positives = [
#     "Hey Rot, what’s the weather like today?",
#     "Yo @rotrot_botbot! What’s up? Got any fun jokes for me?",
#     "RotBot, can you help me with my homework? I need to find the capital of Mongolia.",
#     "Rot, do you ever wonder what it’s like to be human?",
#     "Hey @rotrot_botbot, how do I fix this error in Python? It says 'IndexError: list index out of range.'",
#     "RotBot, I’m feeling kinda lost lately. Any wise words for me?",
#     "Rot, who would win in a fight: Batman or Iron Man?",
#     "@rotrot_botbot, I need to buy a new phone. What’s the best option under $500?",
#     "Rot, why is the sky blue?",
#     "Yo RotBot, got any tips for leveling up quickly in Elden Ring?",
#     "Hey RotBot, can you recommend a good book to read?",
#     "@rotrot_botbot, what’s 123 multiplied by 456?",
#     "Rot, can you tell me a random fun fact about space?",
#     "Hey RotBot, how do I make the perfect cup of coffee?",
#     "@rotrot_botbot, can you translate 'Hello' into Japanese?",
#     "Rot, do you think AI will ever take over the world?",
#     "Hey RotBot, what’s the best way to lose weight without dieting?",
#     "Rot, can you help me find a good recipe for chocolate cake?",
#     "@rotrot_botbot, what’s the most visited city in the world?",
#     "RotBot, how do I write a polite email to my professor?",
#     "Hey Rot, what time is it in Tokyo right now?",
#     "@rotrot_botbot, can you generate a random number for me?",
#     "Rot, how do I fix my Wi-Fi when it keeps disconnecting?",
#     "RotBot, can you summarize the latest news for me?",
#     "Hey @rotrot_botbot, what’s the difference between a comet and an asteroid?",
#     "Rot, what’s your favorite type of music?",
#     "Hey RotBot, I need a workout plan for beginners. Any suggestions?",
#     "@rotrot_botbot, who invented the light bulb?",
#     "Rot, can you recommend a good movie for a Friday night?",
#     "RotBot, what’s the best way to start learning a new language?"
# ]

# genuine_negatives = [
#     'Lha, rotbot evil mana',
#     'uh oh',
#     'Stres dia',
#     'jujur gw ga confident dia bisa',
#     'kalo mau switch ke coder atau classic dulu',
#     'bro casted a spell on u',
#     "everything is pretty much in place now gw perlu refactor sm benerin hal",
#     "coder plain gpt, classic rotbot sebelum lobotomy ini",
#     "versi lebih upbeat yang dulu",
#     "versi lebih lobotomized",
#     "Ayyo?",
#     "hmm",
#     "Okey tbf",
#     "Udang neocaridina klo campur warnany",
#     "Balik ke wild version",
#     "Jdi brown clear",
#     "Jdi kyk dua udang ssr dicampur jdi common",
#     "Julian pke they / them pronouns tah",
#     "Pilih kasih dia",
#     "baru works on penis",
#     "Anjir",
#     "Hayo dimarahin",
#     "Sassy banget anjir",
#     "Sassy man apocalypse starts with rotbot",
#     "wait gw jadi pengen nambah info first name last name",
#     "biar si rot bisa panggil nama gw",
#     "like what rot",
#     "WHAT IS IT ROT I DONT SEE IT",
#     "rot please what do you want me to do",
#     "ok tbf gw lagi refactor codenya rotbot sih...",
#     "bro imagine getting gentle paranted by your own creation",
#     "Merasa emasculated kah dirimu",
#     "Im not talking to u rot",
#     "Learn to read the room cui",
#     "yea itu kenapa gw perlu data kalian",
#     "its paradoxical, but it works",
#     "I see, org minang kurang go global",
#     "Imut bgt",
#     "jir itu designed by nya kenapa gaada kalian",
#     "Mrk tarok at the end kah gituan",
#     "wait wtf sejak kapan ada logo",
#     "Takut gamuat",
#     "jir wtff",
#     "mini hx itu gimana buatnya",
#     "Ggs guys",
#     "Sim focus topic is officially dissolved",
#     "wise words",
#     "i have to stop him from using semicolons",
#     "Huh???",
#     "why dont you think this is a good idea",
#     "and that better option is?",
#     "Rotbot universe",
#     "anj",
#     "Itu pcb nya accoustic 💀",
#     "ternyata 6 microfit 1 header biasa",
#     "Aman lah",
#     "Kn akhirny lubang smua",
#     "Kita yg solder sendiri",
#     "wait why did you do that rot",
#     "A little brain rot is good for the soul; don't take it too seriously."
# ]

# synthetic_negatives = [
#     "Hey everyone, what time are we meeting tomorrow?",
#     "Did anyone finish the assignment? I’m stuck on question 5.",
#     "Happy birthday, Alex! 🎉 Hope you’re having an amazing day!",
#     "Guys, I just saw the funniest meme, you HAVE to check this out 😂.",
#     "Don’t forget, the potluck is this Saturday. What’s everyone bringing?",
#     "Anyone else having trouble with the WiFi, or is it just me?",
#     "Thanks for the ride earlier, Sam. You’re a lifesaver!",
#     "What movie are we watching for the next movie night? Any suggestions?",
#     "OMG, did you guys see the news today? That was wild!",
#     "Let’s plan a game night soon. I miss hanging out with all of you!"
# ]

# embeddings = []

# for text in synthetic_positives:
#     response = openai_client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"  # Example model; adjust as needed
#     )
#     embeddings.append(response.data[0].embedding)  

# embeddings_array = np.array(embeddings)

# average_embedding = np.mean(embeddings_array, axis=0)

# norm = np.linalg.norm(average_embedding)

# if norm > 0:  # To avoid division by zero
#     reference_embedding = average_embedding / norm
# else:
#     reference_embedding = average_embedding  

# positive_response = openai_client.embeddings.create(
#     input='no wrong one rot, try directly searching for tomato cat',
#     model="text-embedding-3-small"  # Example model; adjust as needed
# )

# negative_response = openai_client.embeddings.create(
#     input='No, wrong one Rot. Try directly searching for \'Tomato Cat\'',
#     model="text-embedding-3-small"  # Example model; adjust as needed
# )

# positive_cosine_similarity = np.dot(np.array(positive_response.data[0].embedding), reference_embedding)
# negative_cosine_similarity = np.dot(np.array(negative_response.data[0].embedding), reference_embedding)
