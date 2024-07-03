# POLLI-SHOW

**POLLI-SHOW** is a Discord bot for generating AI art using Pollinations.

## Features

- Generate AI art with specified prompts and styles
- Schedule automatic art generation
- Manage saved images to keep only the latest 500

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Tolerable/POLLI-SHOW.git
    cd POLLI-SHOW
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your environment variables:
    ```plaintext
    OPENAI_API_KEY=<your_openai_api_key>
    ARTSHOWPOLLI_DISCORD_BOT_TOKEN=<your_discord_bot_token>
    ```

4. Run the bot:
    ```sh
    python POLLI-SHOW.py
    ```

## Usage

- Use `/polliart <prompt> <style1> <style2>` to generate an image with specified styles.
- Use `/pollishow <interval> <file_number>` to manage scheduled prompts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
