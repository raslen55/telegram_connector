import asyncio
import re
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.errors import FloodWaitError, ChannelPrivateError, UserAlreadyParticipantError, UsernameInvalidError, UsernameNotOccupiedError
from telethon.tl.types import Channel, User, Channel, Chat
import multiprocessing
from functools import partial
import argparse
import json
import random
import signal
import os
from datetime import datetime
from colorama import init, Fore, Back, Style
import sys
import ipaddress

init(autoreset=True)

PURPLE_BLUE = '\033[38;2;100;100;255m'
LIGHT_PURPLE = '\033[38;2;200;180;255m'
BOLD_WHITE = '\033[1;37m'

def print_info(message):
    print(f"{PURPLE_BLUE}ℹ {BOLD_WHITE}{message}")

def print_success(message):
    print(f"{LIGHT_PURPLE}✔ {BOLD_WHITE}{message}")

def print_warning(message):
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {BOLD_WHITE}{message}")

def print_error(message):
    print(f"{Fore.RED}✘ {message}")

def print_header(message):
    print(f"\n{PURPLE_BLUE}{Style.BRIGHT}{message}")
    print(f"{PURPLE_BLUE}{'-' * len(message)}{Style.RESET_ALL}")

def print_subheader(message):
    print(f"\n{LIGHT_PURPLE}{Style.BRIGHT}{message}")
    print(f"{LIGHT_PURPLE}{'-' * len(message)}{Style.RESET_ALL}")

def banner():
    print(f"""
          
{Fore.BLUE}{Style.BRIGHT}


                      +++++                      
                    ++{LIGHT_PURPLE}=   +{Style.RESET_ALL}{Fore.BLUE}{Style.BRIGHT}+                     
                    ++{LIGHT_PURPLE}+   ++{Style.RESET_ALL}{Fore.BLUE}{Style.BRIGHT}+                    
                    +++{LIGHT_PURPLE}+++{Style.RESET_ALL}{Fore.BLUE}{Style.BRIGHT}++*                    
                    *+++*+***                    
                     ********                    
                   {LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT}**********                   
                  **{LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT} *********                  
                 ***{LIGHT_PURPLE}##{Fore.BLUE}{Style.BRIGHT}**********                 
               *****{LIGHT_PURPLE}###{Fore.BLUE}{Style.BRIGHT}***********{LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT}              
           *********{LIGHT_PURPLE}####{Fore.BLUE} ******{LIGHT_PURPLE}########{Fore.BLUE}{Style.BRIGHT}          
 ++{LIGHT_PURPLE}+{Fore.BLUE}{Style.BRIGHT}++**************{LIGHT_PURPLE}###   #######{Fore.BLUE}{Style.BRIGHT}  *******++{LIGHT_PURPLE}++{Fore.BLUE}{Style.BRIGHT}++ 
+{LIGHT_PURPLE}++  +{Fore.BLUE}{Style.BRIGHT}**************{LIGHT_PURPLE}#       ##{Fore.BLUE}{Style.BRIGHT} *************  +{LIGHT_PURPLE}{Fore.BLUE}{Style.BRIGHT}++
++{LIGHT_PURPLE}+   +{Fore.BLUE}{Style.BRIGHT}***********  {LIGHT_PURPLE}#       #{Fore.BLUE}{Style.BRIGHT}*************+*  +{LIGHT_PURPLE}{Fore.BLUE}{Style.BRIGHT}++
 +++{LIGHT_PURPLE}++{Fore.BLUE}{Style.BRIGHT}******** {LIGHT_PURPLE}########   ###{Fore.BLUE}{Style.BRIGHT}*************++{LIGHT_PURPLE}++{Fore.BLUE}{Style.BRIGHT}++ 
        {LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT}**{LIGHT_PURPLE}####{Fore.BLUE}{Style.BRIGHT}****** {LIGHT_PURPLE}###{Fore.BLUE}{Style.BRIGHT}***********          
              ************{LIGHT_PURPLE}###{Fore.BLUE}{Style.BRIGHT}*****               
                 **********{LIGHT_PURPLE}##{Fore.BLUE}{Style.BRIGHT}***                 
                  ********* {LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT}**                  
                   ********* *                   
                    ******** {LIGHT_PURPLE}#{Fore.BLUE}{Style.BRIGHT}                   
                    *********                    
                    **+{LIGHT_PURPLE}**{Fore.BLUE}{Style.BRIGHT}+***                    
                    *+{LIGHT_PURPLE}+   +{Fore.BLUE}{Style.BRIGHT}++                    
                     +{LIGHT_PURPLE}+   +{Fore.BLUE}{Style.BRIGHT}++                    
                      ++{LIGHT_PURPLE}+{Fore.BLUE}{Style.BRIGHT}++                      

                    


   
{Style.RESET_ALL}
""")

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print_info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

# Extract Telegram channel links from messages
def extract_channel_links(text):
    """Extract Telegram channel links from text"""
    if not isinstance(text, str):
        return []
    
    links = []
    # Match t.me links
    tme_links = re.findall(r'(?:https?://)?t\.me/([a-zA-Z]\w{3,30}[a-zA-Z\d])', text)
    links.extend([f"@{username}" for username in tme_links])
    
    # Match direct @ mentions
    mentions = re.findall(r'@([a-zA-Z]\w{3,30}[a-zA-Z\d])', text)
    links.extend([f"@{username}" for username in mentions])
    
    # Match channel links
    channel_links = re.findall(r'(?:https?://)?(?:www\.)?telegram\.me/([a-zA-Z]\w{3,30}[a-zA-Z\d])', text)
    links.extend([f"@{username}" for username in channel_links])
    
    return list(set(links))  # Remove duplicates

# Clean and format channel links
def clean_link(link):
    if not link or not isinstance(link, str):
        return None
    
    link = link.split(')')[0].strip()
    
    if re.match(r'^[a-zA-Z0-9_]{5,}$', link):
        return link
    
    match = re.search(r't\.me/(?:joinchat/)?([a-zA-Z0-9_-]+)', link)
    if match:
        username_or_hash = match.group(1)
        
        if 'joinchat' in link:
            return f'https://t.me/joinchat/{username_or_hash}'
        
        return username_or_hash
    
    return None

# Manage discovered channels
class ChannelManager:
    def __init__(self):
        self.discovered_channels = set()
        self.processed_channels = set()
        self.channel_sources = {}
    
    def add_channel(self, link, source_channel=None):
        """Add a channel to be processed"""
        if not isinstance(link, str):
            return
        
        # Clean up the link
        link = link.strip()
        if not link:
            return
        
        # Validate username format
        username = link[1:] if link.startswith('@') else link
        if not re.match(r'^[a-zA-Z]\w{3,30}[a-zA-Z\d]$', username):
            return
        
        # Add to discovered channels
        self.discovered_channels.add(f"@{username}")
        if source_channel:
            self.channel_sources[f"@{username}"] = source_channel
    
    def get_next_channel(self):
        """Get the next unprocessed channel"""
        unprocessed = self.discovered_channels - self.processed_channels
        return next(iter(unprocessed)) if unprocessed else None
    
    def mark_as_processed(self, link):
        """Mark a channel as processed"""
        if isinstance(link, str):
            link = link.strip()
            username = link[1:] if link.startswith('@') else link
            self.processed_channels.add(f"@{username}")
    
    def has_unprocessed_channels(self):
        """Check if there are unprocessed channels"""
        return bool(self.discovered_channels - self.processed_channels)
    
    def get_new_channels(self):
        """Get newly discovered channels"""
        return self.discovered_channels - self.processed_channels

# Join channel by url
async def join_channel(client, channel_manager, link):
    """Join a Telegram channel"""
    try:
        # Clean up the link format
        if link.startswith('https://t.me/'):
            username = link.split('/')[-1]
            link = f"@{username}"
        elif link.startswith('https://telegram.me/'):
            username = link.split('/')[-1]
            link = f"@{username}"
        elif not link.startswith('@'):
            link = f"@{link}"
        
        # Validate username format
        username = link[1:] if link.startswith('@') else link
        if not re.match(r'^[a-zA-Z]\w{3,30}[a-zA-Z\d]$', username):
            print_warning(f"Invalid username format: {link}")
            return False
        
        try:
            await client(JoinChannelRequest(link))
            print_success(f"Successfully joined channel: {link}")
            return True
        except UserAlreadyParticipantError:
            print_info(f"Already a member of channel: {link}")
            return True
        except (UsernameInvalidError, UsernameNotOccupiedError):
            print_warning(f"Invalid or non-existent channel: {link}")
            return False
        except FloodWaitError as e:
            print_warning(f"FloodWaitError while joining {link}: {e}")
            await asyncio.sleep(e.seconds)
            return False
        except Exception as e:
            print_error(f"Error joining channel {link}: {str(e)}")
            return False
    except Exception as e:
        print_error(f"Unexpected error joining channel {link}: {str(e)}")
        return False

# Load configuration
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print_error(f"Config file not found at {config_path}")
        return create_default_config(config_path)
    except json.JSONDecodeError:
        print_error(f"Invalid JSON in config file {config_path}")
        return create_default_config(config_path)

# Create a default config file, if no config present 
def create_default_config(config_path):
    default_config = {
        "initial_channel_links": [],
        "message_keywords": [],
        "batch_size": 100,
        "api_id": "",
        "api_hash": "",
        "phone_number": ""
    }
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print_success(f"Default config file created at {config_path}")
    print_info("Please edit this file with your channel links and keywords.")
    return default_config

# Home made sentiment lexicon 
class CybersecuritySentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.cybersecurity_lexicon = {
            'vulnerability': 2.0,
            'exploit': -3.0,
            'patch': 2.0,
            'hack': -2.0,
            'secure': 3.0,
            'breach': -4.0,
            'protect': 3.0,
            'malware': -3.0,
            'ransomware': -4.0,
            'encryption': 2.0,
            'backdoor': -3.0,
            'firewall': 2.0,
            'phishing': -3.0,
            'authentication': 2.0,
            'threat': -2.0,
            'zero-day': -4.0,
            'security': 1.0,
            'attack': -2.0,
            'defense': 2.0,
            'compromise': -3.0
        }
        self.sia.lexicon.update(self.cybersecurity_lexicon)

    def polarity_scores(self, text):
        return self.sia.polarity_scores(text)

# Global variables
current_batch = []
batch_counter = 1

# keyboard interrupt (Ctrl+C)
def signal_handler(sig, frame):
    global current_batch, batch_counter
    print_warning(f"\nKeyboard interrupt received. Saving current batch and exiting...")
    save_current_batch(current_batch, batch_counter)
    exit(0)

# Save current batch to CSV
def save_current_batch(batch, batch_counter):
    """Save the current batch of messages to a CSV file"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        # Convert batch data to DataFrame
        df = pd.DataFrame(batch)
        
        # Clean and normalize data
        df['Message'] = df['Message'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else str(x))
        df['Channel Name'] = df['Channel Name'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else str(x))
        df['Affiliated Channel'] = df['Affiliated Channel'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else str(x))
        
        # Save to CSV with proper encoding
        filename = os.path.join(data_dir, f'telegram_scraped_messages_batch_{batch_counter}.csv')
        df.to_csv(filename, index=False, encoding='utf-8')
        print_success(f"Saved batch {batch_counter} with {len(batch)} messages to {filename}")
        return True
    except Exception as e:
        print_error(f"Error saving batch {batch_counter}: {str(e)}")
        return False

async def scrape_messages(client, entity, message_depth, keywords, channel_manager):
    """Scrape messages from a channel"""
    messages = []
    try:
        async for message in client.iter_messages(entity, limit=message_depth):
            if not message or not message.text:
                continue
            
            # Check if message contains any keywords
            if not any(keyword.lower() in message.text.lower() for keyword in keywords):
                continue
            
            # Extract message data
            message_data = {
                'Sender ID': str(message.sender_id) if message.sender_id else 'Unknown',
                'Date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
                'Message': message.text,
                'Channel Name': entity.title if hasattr(entity, 'title') else str(entity.id),
                'Affiliated Channel': None  # Will be set by the batch processor
            }
            messages.append(message_data)
            
            # Extract and add new channels
            new_links = extract_channel_links(message.text)
            for link in new_links:
                channel_manager.add_channel(link, source_channel=entity.title if hasattr(entity, 'title') else str(entity.id))
        
        if messages:
            print_success(f"Found {len(messages)} relevant messages in {entity.title if hasattr(entity, 'title') else str(entity.id)}")
        return messages
    
    except Exception as e:
        print_error(f"Error scraping messages: {str(e)}")
        return []

async def process_single_channel(client, channel_manager, link, message_depth, keywords):
    """Process a single Telegram channel"""
    try:
        # Clean up and validate the link
        if isinstance(link, str):
            link = link.strip()
            if not link:
                return []
        else:
            print_warning(f"Invalid link type: {type(link)}")
            return []
        
        join_success = await retry_with_backoff(join_channel(client, channel_manager, link))
        if join_success:
            try:
                entity = await client.get_entity(link)
                if hasattr(entity, 'title'):
                    print_info(f"Scraping messages from: {entity.title}")
                else:
                    print_info(f"Scraping messages from: {link}")
                
                messages = await scrape_messages(client, entity, message_depth, keywords, channel_manager)
                if messages:
                    print_success(f"Successfully scraped {len(messages)} messages from {entity.title if hasattr(entity, 'title') else link}")
                return messages
            except ValueError as e:
                if "Could not find the input entity" in str(e):
                    print_warning(f"Channel not found: {link}")
                else:
                    print_error(f"Failed to process entity {link}: {str(e)}")
                channel_manager.mark_as_processed(link)
                return []
            except Exception as e:
                print_error(f"Failed to process entity {link}: {str(e)}")
                channel_manager.mark_as_processed(link)
                return []
        else:
            print_warning(f"Could not join channel: {link}")
            channel_manager.mark_as_processed(link)
            return []
    except Exception as e:
        print_error(f"Error processing channel {link}: {str(e)}")
        channel_manager.mark_as_processed(link)
        return []

class BatchProcessor:
    def __init__(self, batch_size=1000, cybersecurity_sia=None):
        self.batch = []
        self.batch_size = batch_size
        self.batch_counter = 1
        self.total_messages = 0
        self.cybersecurity_sia = cybersecurity_sia or CybersecuritySentimentAnalyzer()
        self.data_dir = "data"
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.all_messages_df = pd.DataFrame(columns=[
            'Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment',
            'Channel Name', 'Affiliated Channel', 'URLs', 'IP_Addresses', 'Hashes'
        ])

    def add_messages(self, messages, channel_name, affiliated_channel):
        """Add messages to the current batch"""
        if not messages:
            return

        for message in messages:
            # Extract IOCs
            urls = extract_urls(message['Message'])
            ips = extract_ips(message['Message'])
            hashes = extract_hashes(message['Message'])
            
            # Calculate sentiment
            sentiment = self.cybersecurity_sia.polarity_scores(message['Message'])
            
            message_data = {
                'Sender ID': message['Sender ID'],
                'Date': message['Date'],
                'Message': message['Message'],
                'Sentiment': sentiment,
                'Compound_Sentiment': sentiment['compound'],
                'Channel Name': message['Channel Name'],
                'Affiliated Channel': message['Affiliated Channel'],
                'URLs': urls,
                'IP_Addresses': ips,
                'Hashes': hashes
            }
            self.batch.append(message_data)
            self.total_messages += 1

            if len(self.batch) >= self.batch_size:
                self.save_batch()

    def save_batch(self):
        """Save the current batch to a CSV file"""
        if not self.batch:
            return
        
        try:
            # Convert batch to DataFrame
            batch_df = pd.DataFrame(self.batch)
            
            # Convert lists to strings for CSV storage
            list_columns = ['URLs', 'IP_Addresses', 'Hashes']
            for col in list_columns:
                if col in batch_df.columns:
                    batch_df[col] = batch_df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))
            
            # Save batch
            filename = os.path.join(self.data_dir, f'telegram_scraped_messages_batch_{self.batch_counter}.csv')
            batch_df.to_csv(filename, index=False, encoding='utf-8')
            print_success(f"Saved batch {self.batch_counter} with {len(self.batch)} messages to {filename}")
            
            # Update complete dataset
            self.all_messages_df = pd.concat([self.all_messages_df, batch_df], ignore_index=True)
            
            # Reset batch
            self.batch = []
            self.batch_counter += 1
        except Exception as e:
            print_error(f"Error saving batch {self.batch_counter}: {str(e)}")

    def generate_final_report(self):
        """Generate the final report and save complete dataset"""
        # Save any remaining messages
        if self.batch:
            self.save_batch()
        
        if len(self.all_messages_df) == 0:
            print_warning("No messages collected. Skipping final report generation.")
            return
        
        try:
            # Save complete dataset
            final_file = os.path.join(self.data_dir, 'complete_dataset.csv')
            self.all_messages_df.to_csv(final_file, index=False, encoding='utf-8')
            print_success(f"Saved complete dataset with {len(self.all_messages_df)} messages to {final_file}")
            
            # Generate threat report
            report_file = os.path.join(self.data_dir, 'threat_report.json')
            
            # Basic statistics
            stats = {
                'total_messages': len(self.all_messages_df),
                'unique_channels': self.all_messages_df['Channel Name'].nunique(),
                'date_range': {
                    'start': self.all_messages_df['Date'].min(),
                    'end': self.all_messages_df['Date'].max()
                },
                'threat_levels': {
                    'high': len(self.all_messages_df[self.all_messages_df['Compound_Sentiment'] <= -0.5]),
                    'medium': len(self.all_messages_df[(self.all_messages_df['Compound_Sentiment'] > -0.5) & 
                                                     (self.all_messages_df['Compound_Sentiment'] < -0.2)]),
                    'low': len(self.all_messages_df[self.all_messages_df['Compound_Sentiment'] >= -0.2])
                },
                'ioc_summary': {
                    'total_urls': self.all_messages_df['URLs'].str.count(',').sum() + len(self.all_messages_df[self.all_messages_df['URLs'] != '']),
                    'total_ips': self.all_messages_df['IP_Addresses'].str.count(',').sum() + len(self.all_messages_df[self.all_messages_df['IP_Addresses'] != '']),
                    'total_hashes': self.all_messages_df['Hashes'].str.count(',').sum() + len(self.all_messages_df[self.all_messages_df['Hashes'] != ''])
                }
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)
            print_success(f"Generated threat report: {report_file}")
            
        except Exception as e:
            print_error(f"Error generating final report: {str(e)}")

    def finalize(self):
        """Finalize processing and generate reports"""
        self.generate_final_report()

    def __del__(self):
        """Ensure all data is saved when object is destroyed"""
        if self.batch:
            self.save_batch()

async def retry_with_backoff(coroutine, max_retries=5, base_delay=1, max_delay=60):
    retries = 0
    while True:
        try:
            return await coroutine
        except FloodWaitError as e:
            if retries >= max_retries:
                raise
            delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
            print_warning(f"FloodWaitError encountered. Retrying in {delay:.2f} seconds. (Attempt {retries + 1}/{max_retries})")
            await asyncio.sleep(delay)
            retries += 1
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            raise

def extract_urls(text):
    """Extract URLs from text"""
    if not isinstance(text, str):
        return []
    
    # URL pattern matching common protocols
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)

def extract_ips(text):
    """Extract IP addresses from text"""
    if not isinstance(text, str):
        return []
    
    # IPv4 pattern
    ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    # IPv6 pattern
    ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'
    
    ips = re.findall(ipv4_pattern, text)
    ips.extend(re.findall(ipv6_pattern, text))
    return [ip for ip in ips if is_valid_ip(ip)]

def extract_hashes(text):
    """Extract common hash values (MD5, SHA1, SHA256) from text"""
    if not isinstance(text, str):
        return []
    
    # Hash patterns
    md5_pattern = r'\b[a-fA-F0-9]{32}\b'
    sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
    sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
    
    hashes = []
    hashes.extend(re.findall(md5_pattern, text))
    hashes.extend(re.findall(sha1_pattern, text))
    hashes.extend(re.findall(sha256_pattern, text))
    return hashes

def is_valid_ip(ip):
    """Validate IP address"""
    try:
        if ':' in ip:  # IPv6
            ipaddress.IPv6Address(ip)
        else:  # IPv4
            ipaddress.IPv4Address(ip)
        return True
    except ValueError:
        return False

async def run_scraper(config, message_depth, channel_depth):
    try:
        start_time = datetime.now()  # Initialize start_time
        
        api_id = int(config.get('api_id'))
        api_hash = config.get('api_hash')
        phone_number = config.get('phone_number')
        initial_channels = config.get('initial_channel_links', [])
        keywords = config.get('message_keywords', [])
        batch_size = config.get('batch_size', 100)

        if not all([api_id, api_hash, phone_number]):
            print_error("Missing required configuration. Please check config.json")
            return

        print_info("Connecting to Telegram...")
        client = TelegramClient('telehunting_session', api_id, api_hash)
        
        try:
            await client.connect()
            if not await client.is_user_authorized():
                print_info("First time login, sending code request...")
                await client.send_code_request(phone_number)
                code = input('Enter the code you received: ')
                await client.sign_in(phone_number, code)
            print_success("Successfully connected to Telegram!")
        except Exception as e:
            print_error(f"Authentication error: {str(e)}")
            return

        # Initialize managers and processors
        channel_manager = ChannelManager()
        batch_processor = BatchProcessor(batch_size=batch_size)

        # Add initial channels
        for link in initial_channels:
            channel_manager.add_channel(link)

        print_info("\nProcessing Initial Channels")
        print_info("-----------------------")
        
        try:
            # Process initial channels first
            for link in initial_channels:
                try:
                    messages = await process_single_channel(client, channel_manager, link, message_depth, keywords)
                    if messages:
                        batch_processor.add_messages(messages, link, None)
                        print_success(f"Successfully processed {len(messages)} messages from {link}")
                    else:
                        print_warning(f"No messages found in channel: {link}")
                except Exception as e:
                    print_error(f"Error processing channel {link}: {str(e)}")
                    continue

            # Process discovered channels up to the specified depth
            depth = 1
            while depth < channel_depth and channel_manager.has_unprocessed_channels():
                print_info(f"\nProcessing Depth {depth}")
                print_info("-" * (18 + len(str(depth))))
                
                channels_at_depth = list(channel_manager.get_new_channels())
                for link in channels_at_depth:
                    try:
                        messages = await process_single_channel(client, channel_manager, link, message_depth, keywords)
                        if messages:
                            source = channel_manager.channel_sources.get(link)
                            batch_processor.add_messages(messages, link, source)
                            print_success(f"Successfully processed {len(messages)} messages from {link}")
                        else:
                            print_warning(f"No messages found in channel: {link}")
                    except Exception as e:
                        print_error(f"Error processing channel {link}: {str(e)}")
                        continue
                
                depth += 1

        except Exception as e:
            print_error(f"Error during channel processing: {str(e)}")
        finally:
            # Generate final report
            print_info("\nGenerating Final Report")
            print_info("-----------------------")
            batch_processor.finalize()
            
            # Disconnect client
            await client.disconnect()
            print_success(f"Completed in {datetime.now() - start_time}")

    except Exception as e:
        print_error(f"Fatal error: {str(e)}")
        if 'client' in locals():
            await client.disconnect()

if __name__ == "__main__":
    banner()
    ensure_nltk_data()

    parser = argparse.ArgumentParser(description='Telegram Channel Message Scraper')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--message-depth', type=int, help='Number of messages to scrape per channel')
    parser.add_argument('--channel-depth', type=int, help='Depth of channel discovery')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()
    
    config = load_config(args.config)
    if not config:
        sys.exit(1)

    message_depth = args.message_depth or config.get('message_depth', 1000)
    channel_depth = args.channel_depth or config.get('channel_depth', 2)

    asyncio.run(run_scraper(config, message_depth, channel_depth))
