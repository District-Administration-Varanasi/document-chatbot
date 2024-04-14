# Question & Answer PDF using Gemini and langchain 
Made by: [Aleen Dhar](https://www.linkedin.com/in/aleendhar/)


## Working:
![chrome_oAcmNTpTRn](https://github.com/AleenDhar/document-chatbot/assets/86429480/f6894928-41e8-4dc7-992c-a497fcc8df56)

## Environment Setup:

1. make a .env file and copy paste the following code
```bash

```

2. get AGENT_MAILBOX_KEY from https://agentverse.ai/mailroom
3. get HUGGING_FACE_ACCESS_TOKEN from https://huggingface.co/settings/tokens
4. get gemini_api_key from https://aistudio.google.com/app/u/2/apikey
5. get elevenlabs_api_key from https://elevenlabs.io/  and sign up, click on your icon and choose Profile + API key
6. eleven_voice_id from https://elevenlabs.io/app/voice-lab.
7. get cloudinary_cloud_name, cloudinary_api_key,cloudinary_api_secret from https://console.cloudinary.com/pm/c-2975fa68853eb272867546601d974b/getting-started



## Setup
1. In the main directory install all dependencies

    ```bash
    python -m poetry install
    ```


## Running The Main Script

To run the project, use the command:

    ```
    cd src
    pyhton -m poetry run python main.py
    ```

