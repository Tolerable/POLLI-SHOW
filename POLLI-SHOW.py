import discord
from discord.ext import commands, tasks
from discord import app_commands
import os
import base64
import io
from PIL import Image
import datetime
import aiohttp
import random
import sqlite3
import asyncio
import glob
import re
import openai
import requests
import urllib.parse

# Initialize Discord bot with intents
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True
intents.message_content = True
intents.reactions = True
intents.emojis_and_stickers = True
intents.integrations = True
intents.webhooks = True
intents.invites = False
intents.voice_states = False
intents.presences = True
intents.typing = False

bot = commands.Bot(command_prefix='/', intents=intents)

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client with API key
client = openai

# Set your OpenAI API key
client.api_key = os.getenv('OPENAI_API_KEY')

# Load styles from ./ASSETS/STYLES.txt
def load_styles():
    style_tags = {}
    with open('./ASSETS/STYLES.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split(':')
                name = parts[0].strip()
                traits = ':'.join(parts[1:]).split('),')
                positive_tags = traits[0].strip().strip('()').split(',')
                negative_tags = traits[1].strip().strip('()').split(',')
                style_tags[name] = (positive_tags, negative_tags)
    return style_tags

style_tags = load_styles()

# Autocomplete function for styles
async def style_autocomplete(interaction: discord.Interaction, current: str):
    return [
        app_commands.Choice(name=style, value=style)
        for style in style_tags.keys()
        if current.lower() in style.lower()
    ]

# Updated command with autocomplete
@bot.tree.command(name="polliart", description="Generate an image with Pollinations")
@app_commands.describe(prompt="The prompt for the image", style1="First style (optional)", style2="Second style (optional)")
@app_commands.autocomplete(style1=style_autocomplete, style2=style_autocomplete)
async def polliart(
    interaction: discord.Interaction,
    prompt: str,
    style1: str = None,
    style2: str = None
):
    try:
        # Validate provided styles
        available_styles = style_tags.keys()
        selected_styles = []
        if style1:
            if style1 in available_styles:
                selected_styles.append(style1)
            else:
                await interaction.response.send_message(f"Invalid style: {style1}. Please choose from the available styles.", ephemeral=True)
                return
        if style2:
            if style2 in available_styles:
                selected_styles.append(style2)
            else:
                await interaction.response.send_message(f"Invalid style: {style2}. Please choose from the available styles.", ephemeral=True)
                return

        # Sanitize input
        sanitized_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)
        print(f"[DEBUG] Received sanitized prompt: {sanitized_prompt}")

        # Check if the prompt is safe
        is_safe, message = mod_check(sanitized_prompt)
        if not is_safe:
            await interaction.response.send_message(message, ephemeral=True)
            return

        await interaction.response.send_message(f"Received prompt: {sanitized_prompt}", ephemeral=True)

        # Combine prompt and styles for the debug message
        full_prompt_debug = f"{sanitized_prompt}, {', '.join([f'{style} style' for style in selected_styles])}"

        # Add user request to the queue
        await user_request_queue.put((sanitized_prompt, selected_styles))
        print(f"[DEBUG] Added prompt to user_request_queue: {full_prompt_debug}")

    except Exception as e:
        print(f"[DEBUG] Error in polliart command: {e}")
        await interaction.response.send_message(f"Error: {e}", ephemeral=True)

# Function to parse prompts
def parse_prompt(prompt):
    return prompt

@bot.event
async def on_raw_reaction_add(payload):
    if str(payload.emoji) == "🚫":
        # Fetch the channel
        if payload.guild_id is None:  # If this is a DM
            channel = await bot.fetch_channel(payload.channel_id)
        else:  # If this is within a guild
            channel = bot.get_channel(payload.channel_id)

        if channel:
            message = await channel.fetch_message(payload.message_id)
            if message.author == bot.user:
                # In DMs, anyone can delete the bot's messages
                if payload.guild_id is None:
                    await message.delete()
                else:
                    # In guilds, check if the user who reacted is an admin
                    guild = bot.get_guild(payload.guild_id)
                    member = guild.get_member(payload.user_id)
                    if member and member.guild_permissions.administrator:
                        await message.delete()
    elif str(payload.emoji) == "🔁":  # Re-generate on 🔁 emoji
        channel = bot.get_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        if message.author == bot.user:
            embed = message.embeds[0]
            prompt_text = embed.description.split("**Prompt:** ")[1].split("\n")[0]
            await generate_new_image(channel, prompt_text)

# Function to manage the number of saved images
def manage_saved_images(directory, max_files=500):
    files = glob.glob(os.path.join(directory, '*.png'))
    file_count = len(files)
    if file_count > max_files:
        files.sort(key=os.path.getctime)
        files_to_remove = files[:-max_files]
        for file in files_to_remove:
            os.remove(file)
            print(f"[DEBUG] Removed old image: {file}")
        print(f"[DEBUG] Removed {len(files_to_remove)} old images. Current file count: {len(files) - len(files_to_remove)}")
    else:
        print(f"[DEBUG] No images removed. Current file count: {file_count}")

# Function to save image
def save_image(image, path):
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    image_filename = f"{timestamp}.png"
    image_path = os.path.join(path, image_filename)
    image.save(image_path, 'PNG')
    print(f"[DEBUG] Image saved at: {image_path}")

    # Manage saved images to keep only the last 500
    manage_saved_images(path)

    return image_path


# Database functions
def initialize_db():
    conn = sqlite3.connect('prompts.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS used_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    prompt_file TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

def add_used_prompt(prompt, prompt_file):
    conn = sqlite3.connect('prompts.db')
    c = conn.cursor()
    c.execute('INSERT INTO used_prompts (prompt, prompt_file) VALUES (?, ?)', (prompt, prompt_file))
    conn.commit()
    conn.close()

def get_used_prompts(prompt_file):
    conn = sqlite3.connect('prompts.db')
    c = conn.cursor()
    c.execute('SELECT prompt FROM used_prompts WHERE prompt_file = ?', (prompt_file,))
    used_prompts = c.fetchall()
    conn.close()
    return [prompt[0] for prompt in used_prompts]

def clear_used_prompts(prompt_file):
    conn = sqlite3.connect('prompts.db')
    c = conn.cursor()
    c.execute('DELETE FROM used_prompts WHERE prompt_file = ?', (prompt_file,))
    conn.commit()
    conn.close()

# Add a queue for user requests
user_request_queue = asyncio.Queue()

# Function to check prompt with OpenAI Moderations API
def mod_check(prompt):
    try:
        # Use the client's Moderations API endpoint to check the prompt
        response = client.moderations.create(
            input=prompt
        )

        # Extracting the first result from the response
        moderation_result = response.results[0]

        # Determine if the content is safe
        is_safe = not moderation_result.flagged

        # Construct a message based on moderation result
        if is_safe:
            message = "Your prompt is safe."
        else:
            # Extracting flagged categories
            flagged_categories = [cat for cat, flagged in moderation_result.categories.__dict__.items() if flagged]
            message = f"Your prompt has been flagged as potentially unsafe in the following categories: {', '.join(flagged_categories)}. Please try a different prompt."
            print(f"[DEBUG] Prompt flagged as potentially unsafe in the following categories: {', '.join(flagged_categories)}.")
        return is_safe, message
    except Exception as e:
        print(f"An error occurred during moderation check: {e}")
        return False, "An error occurred during the moderation check. Please try again later."


# Scheduler to generate art based on prompts
@tasks.loop(minutes=2)  # Default interval, adjusted later
async def scheduled_prompt():
    print("[DEBUG] Running scheduled_prompt...")
    channel = bot.get_channel(1255353542252691517)  # Replace with your channel ID for ai-art-pollishow
    if channel:
        # Check if there are any user requests in the queue
        if not user_request_queue.empty():
            user_request = await user_request_queue.get()
            prompt_text, selected_styles = user_request
            parsed_prompt = parse_prompt(prompt_text)

            if selected_styles:
                style_keys = selected_styles
            else:
                num_styles_to_sample = min(2, len(style_tags))
                style_keys = random.sample(list(style_tags.keys()), num_styles_to_sample)

            positive_prompts = [', '.join(style_tags[style][0]).format(parsed_prompt) for style in style_keys]
            positive_prompt = f"{parsed_prompt}, {', '.join(positive_prompts)}"

            print(f"[DEBUG] Using user-requested prompt: {positive_prompt} with styles: {style_keys}")

            # Fetch image from Pollinations
            image, _ = await fetch_image_from_pollinations(positive_prompt)
            if image:
                # Save image
                image_path = save_image(image, './SD_IMAGES')

                # Create the embed
                embed = discord.Embed(title=f'Pollinations A.I. Generated Art (User Requested)',
                                      description=f'**Prompt:** {prompt_text}\n'
                                                  f'**Styles:** {", ".join(style_keys)}\n\n',
                                      color=0x5dade2)
                embed.set_image(url='attachment://image.png')

                await channel.send(file=discord.File(image_path, filename='image.png'), embed=embed)
                print(f"[DEBUG] Sent user-requested prompt: {positive_prompt}")
            else:
                print("[DEBUG] No image generated for user request")
            return

        # Load prompts from all selected files
        all_prompts = []
        for prompt_file in prompt_files_to_use:
            try:
                with open(f'./ASSETS/{prompt_file}', 'r') as f:
                    prompts = f.readlines()
                all_prompts.extend([line for line in prompts if line.strip()])
            except Exception as e:
                print(f"[DEBUG] Error reading {prompt_file}: {e}")

        used_prompts = set()
        for prompt_file in prompt_files_to_use:
            used_prompts.update(get_used_prompts(prompt_file))

        available_prompts = [line for line in all_prompts if line not in used_prompts]

        if not available_prompts:
            for prompt_file in prompt_files_to_use:
                clear_used_prompts(prompt_file)
            available_prompts = all_prompts

        selected_prompt = random.choice(available_prompts)
        for prompt_file in prompt_files_to_use:
            add_used_prompt(selected_prompt, prompt_file)

        print(f"[DEBUG] Processing line: {selected_prompt}")
        parts = selected_prompt.strip().split('" "')
        if len(parts) == 3:
            rating, prompt_text, _ = parts[0].strip('"'), parts[1], parts[2].strip('"')
            parsed_prompt = parse_prompt(prompt_text)

            # Ensure we don't sample more styles than available
            num_styles_to_sample = min(2, len(style_tags))
            style_keys = random.sample(list(style_tags.keys()), num_styles_to_sample)
            positive_prompts = [', '.join(style_tags[style][0]).format(parsed_prompt) for style in style_keys]

            positive_prompt = f"{parsed_prompt}, {', '.join(positive_prompts)}"

            # Generate a random seed to ensure unique image requests
            seed = str(random.randint(0, 99999))

            print(f"[DEBUG] Using prompt: {positive_prompt} with seed: {seed}")

            # Fetch image from Pollinations
            image, _ = await fetch_image_from_pollinations(positive_prompt)
            if image:
                # Save image
                image_path = save_image(image, './SD_IMAGES')

                # Create the embed
                embed = discord.Embed(title=f'Pollinations A.I. Generated Art',
                                      description=f'**Prompt:** {prompt_text}\n'
                                                  f'**Styles:** {", ".join([f"{style} style" for style in style_keys])}\n\n',
                                      color=0x5dade2)
                embed.set_image(url='attachment://image.png')

                await channel.send(file=discord.File(image_path, filename='image.png'), embed=embed)
                print(f"[DEBUG] Sent prompt: {positive_prompt} with rating: {rating}")
            else:
                print("[DEBUG] No image generated")
            await asyncio.sleep(10)  # Adjust as needed
        else:
            print("[DEBUG] Line format incorrect, skipping line.")
    else:
        print("[DEBUG] Channel not found.")

# Function to fetch image from Pollinations
async def fetch_image_from_pollinations(prompt):
    # Generate a random seed to ensure unique image requests
    seed = str(random.randint(0, 99999))
    params = {
        "prompt": prompt,
        "nologo": "true",
        "width": "1360",
        "height": "768",
        "seed": seed
    }
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    print(f"[DEBUG] Fetching image with seed: {seed} and URL: {requests.utils.unquote(url)}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            response.raise_for_status()
            content = await response.read()
            image = Image.open(io.BytesIO(content))
            return image, None  # Pollinations doesn't provide PNG info like SD

async def generate_new_image(channel, prompt_text):
    parsed_prompt = parse_prompt(prompt_text)
    num_styles_to_sample = min(2, len(style_tags))
    style_keys = random.sample(list(style_tags.keys()), num_styles_to_sample)
    positive_prompts = [', '.join(style_tags[style][0]).format(prompt=parsed_prompt) for style in style_keys]
    
    positive_prompt = f"{parsed_prompt}, {', '.join(positive_prompts)}"

    print(f"[DEBUG] Re-generating image with prompt: {positive_prompt}")

    image, _ = await fetch_image_from_pollinations(positive_prompt)
    if image:
        image_path = save_image(image, './SD_IMAGES')
        embed = discord.Embed(title=f'Pollinations A.I. Generated Art',
                              description=f'**Prompt:** {prompt_text}\n'
                                          f'**Styles:** {", ".join([f"{style} style" for style in style_keys])}\n\n',
                              color=0x5dade2)
        embed.set_image(url='attachment://image.png')

        await channel.send(file=discord.File(image_path, filename='image.png'), embed=embed)
        print(f"[DEBUG] Sent re-generated prompt: {positive_prompt}")
    else:
        print("[DEBUG] No image generated for re-generation.")

@bot.tree.command(name='purge', description='Clear the bot\'s messages from the channel.')
async def purge(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)  # Acknowledge the command immediately with a deferred ephemeral response

    try:
        print("Purge command invoked. Resetting the sent status of all entries.")

        # Remove bot's messages from the channel
        channel = interaction.channel
        if channel:
            async for message in channel.history(limit=2500):
                if message.author == bot.user:
                    await message.delete()
                    await asyncio.sleep(1)  # Add delay to prevent rate limiting
            print(f"Bot messages in channel '{channel.name}' have been cleared.")
        else:
            print("Channel not found. Cannot clear messages.")

    except Exception as e:
        print(f"Error during purge operation: {e}")

    # Inform the user that the purge is complete
    await interaction.followup.send("Purge completed. All articles are marked as unsent.", ephemeral=True)
    print("Purge operation completed.")

# Pollishow command to manage the scheduled prompts
@app_commands.command(name='pollishow', description='Manage the scheduled prompts')
@app_commands.describe(interval='Interval in minutes for the scheduled prompts', file_number='Specific prompt file number to use (optional)')
async def pollishow(interaction: discord.Interaction, interval: int, file_number: str = None):
    global current_prompt_file_index
    global prompt_files_to_use
    all_files = list(range(1, len(prompt_files) + 1))

    if interval == 0:
        if not scheduled_prompt.is_running():
            await interaction.response.send_message("Scheduler is not running.", ephemeral=True)
            print("[DEBUG] Scheduler is not running.")
        else:
            scheduled_prompt.stop()
            await interaction.response.send_message("Scheduler stopped.", ephemeral=True)
            print("[DEBUG] Scheduler stopped.")
    else:
        if file_number:
            try:
                files_to_use = parse_file_selection(file_number, all_files)
                prompt_files_to_use = [prompt_files[i - 1] for i in files_to_use]
                current_prompt_file_index = 0
                await interaction.response.send_message(f"Scheduler started with an interval of {interval} minutes using {', '.join(prompt_files_to_use)}.", ephemeral=True)
                print(f"[DEBUG] Scheduler started with an interval of {interval} minutes using {', '.join(prompt_files_to_use)}.")
            except ValueError as e:
                await interaction.response.send_message(str(e), ephemeral=True)
                print(f"[DEBUG] {str(e)}")
                return
        else:
            prompt_files_to_use = prompt_files  # Use all files if no specific file number is given
            current_prompt_file_index = 0
            await interaction.response.send_message(f"Scheduler started with an interval of {interval} minutes using all prompt files.", ephemeral=True)
            print(f"[DEBUG] Scheduler started with an interval of {interval} minutes using all prompt files.")

        scheduled_prompt.change_interval(minutes=interval)
        if not scheduled_prompt.is_running():
            scheduled_prompt.start()

def parse_file_selection(selection, all_files):
    selected_files = set()
    ranges = selection.split(',')
    for r in ranges:
        if '-' in r:
            start, end = map(int, r.split('-'))
            selected_files.update(range(start, end + 1))
        else:
            selected_files.add(int(r))
    if not selected_files.issubset(all_files):
        raise ValueError(f"Invalid file selection. Please select from 1 to {max(all_files)}.")
    return sorted(selected_files)

@bot.event
async def on_ready():
    global current_prompt_file_index
    global prompt_files
    global prompt_files_to_use

    initialize_db()

    current_prompt_file_index = 0  # Initialize the index for cycling through prompt files

    # Dynamically load prompt files and sort them numerically
    prompt_files = sorted(
        [os.path.basename(f) for f in glob.glob('./ASSETS/prompts_*.txt')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    prompt_files_to_use = prompt_files  # Default to using all files

    print(f'Bot is ready. Logged in as {bot.user} (ID: {bot.user.id})')
    try:
        if not bot.tree.get_command('pollishow'):
            bot.tree.add_command(pollishow)
        if not bot.tree.get_command('polliart'):
            bot.tree.add_command(polliart)
        await bot.tree.sync()
        print(f"Synced {len(bot.tree.get_commands())} commands.")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

# Get the bot token from the environment variable
bot_token = os.getenv('ARTSHOWPOLLI_DISCORD_BOT_TOKEN')

if bot_token is None:
    print("Bot token not found. Please set the ARTSHOWPOLLI_DISCORD_BOT_TOKEN environment variable.")
else:
    bot.run(bot_token)
