import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import glob

class ThreatIntelProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.combined_df = None
        self.ensure_nltk_data()
        self.sia = SentimentIntensityAnalyzer()
        
    def ensure_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('vader_lexicon')
    
    def load_data(self):
        """Load data from CSV files in the data directory"""
        try:
            # Look for CSV files in the data directory
            data_dir = 'data'
            csv_files = glob.glob(os.path.join(data_dir, 'telegram_scraped_messages_batch_*.csv'))
            
            if not csv_files:
                # Try loading complete dataset if batch files not found
                complete_dataset = os.path.join(data_dir, 'complete_dataset.csv')
                if os.path.exists(complete_dataset):
                    print(f"Loading complete dataset: {complete_dataset}")
                    self.combined_df = pd.read_csv(complete_dataset)
                    return True
                return False
            
            print(f"Found {len(csv_files)} batch files to process")
            all_dfs = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    print(f"Loaded {len(df)} messages from {os.path.basename(file)}")
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue
            
            if not all_dfs:
                print("No valid data files found")
                return False
            
            self.combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Total messages loaded: {len(self.combined_df)}")
            
            # Save combined dataset
            complete_dataset = os.path.join(data_dir, 'complete_dataset.csv')
            self.combined_df.to_csv(complete_dataset, index=False)
            print(f"Saved complete dataset to: {complete_dataset}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a text using VADER"""
        if not isinstance(text, str):
            return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        return self.sia.polarity_scores(text)

    def process_sentiment(self):
        """Process sentiment for all messages"""
        if self.combined_df is None or len(self.combined_df) == 0:
            return
        
        # Initialize sentiment columns
        self.combined_df['Sentiment'] = None
        self.combined_df['Compound_Sentiment'] = 0.0
        self.combined_df['Threat_Level'] = 'Low'
        
        print("Processing sentiment analysis...")
        # Process messages in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(self.combined_df), chunk_size):
            chunk = self.combined_df.iloc[i:i+chunk_size]
            sentiments = chunk['Message'].apply(self.analyze_sentiment)
            self.combined_df.loc[chunk.index, 'Sentiment'] = sentiments
            self.combined_df.loc[chunk.index, 'Compound_Sentiment'] = sentiments.apply(lambda x: x['compound'])
        
        # Categorize messages
        self.combined_df['Threat_Level'] = self.combined_df['Compound_Sentiment'].apply(
            lambda x: 'High' if x <= -0.5 else
                     'Medium' if -0.5 < x <= -0.2 else
                     'Low'
        )
        
        # Print sentiment distribution
        threat_dist = self.combined_df['Threat_Level'].value_counts()
        print("\nThreat Level Distribution:")
        for level in ['High', 'Medium', 'Low']:
            count = threat_dist.get(level, 0)
            print(f"- {level}: {count} messages")

    def clean_data(self):
        """Clean and normalize the data"""
        if self.combined_df is None:
            return False
        
        try:
            # Initialize required columns if they don't exist
            required_columns = ['Message', 'Channel Name', 'Affiliated Channel', 'Date', 'Sender ID']
            for col in required_columns:
                if col not in self.combined_df.columns:
                    self.combined_df[col] = None
            
            # Convert date strings to datetime with error handling
            self.combined_df['Date'] = pd.to_datetime(self.combined_df['Date'], errors='coerce')
            
            # Drop rows with invalid dates
            invalid_dates = self.combined_df['Date'].isna()
            if invalid_dates.any():
                print(f"Warning: Removed {invalid_dates.sum()} rows with invalid dates")
                self.combined_df = self.combined_df[~invalid_dates].copy()
            
            # Clean text fields
            text_columns = ['Message', 'Channel Name', 'Affiliated Channel']
            for col in text_columns:
                if col in self.combined_df.columns:
                    self.combined_df[col] = self.combined_df[col].apply(
                        lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else str(x)
                    )
            
            # Initialize IOC columns
            self.combined_df['URLs'] = self.combined_df['Message'].apply(lambda x: [])
            self.combined_df['IP_Addresses'] = self.combined_df['Message'].apply(lambda x: [])
            self.combined_df['Hashes'] = self.combined_df['Message'].apply(lambda x: [])
            
            # Clean message text
            self.combined_df['Clean_Message'] = self.combined_df['Message'].apply(self.clean_text)
            
            # Process sentiment
            print("Processing sentiment analysis...")
            self.process_sentiment()
            
            # Extract IOCs
            print("Extracting indicators of compromise...")
            self.combined_df['URLs'] = self.combined_df['Message'].apply(self.extract_urls)
            self.combined_df['IP_Addresses'] = self.combined_df['Message'].apply(self.extract_ips)
            self.combined_df['Hashes'] = self.combined_df['Message'].apply(self.extract_hashes)
            
            # Normalize channel names
            self.combined_df['Channel Name'] = self.combined_df['Channel Name'].str.lower()
            
            return True
        except Exception as e:
            print(f"Error cleaning data: {str(e)}")
            return False
    
    def clean_text(self, text):
        """Clean text data"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        return text
    
    def extract_urls(self, text):
        """Extract URLs from text"""
        if not isinstance(text, str):
            return []
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def extract_ips(self, text):
        """Extract IP addresses from text"""
        if not isinstance(text, str):
            return []
        
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        return re.findall(ip_pattern, text)
    
    def extract_hashes(self, text):
        """Extract potential MD5/SHA hashes from text"""
        if not isinstance(text, str):
            return []
        
        hash_pattern = r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b'
        return re.findall(hash_pattern, text)
    
    def analyze_threats(self):
        """Analyze threats and generate insights"""
        if self.combined_df is None:
            return None
        
        # Handle date analysis safely
        date_min = self.combined_df['Date'].min()
        date_max = self.combined_df['Date'].max()
        
        date_range = {
            'start': date_min.strftime('%Y-%m-%d') if pd.notna(date_min) else 'Unknown',
            'end': date_max.strftime('%Y-%m-%d') if pd.notna(date_max) else 'Unknown'
        }
        
        analysis = {
            'total_messages': len(self.combined_df),
            'unique_channels': self.combined_df['Channel Name'].nunique(),
            'date_range': date_range,
            'threat_levels': {
                'high': len(self.combined_df[self.combined_df['Threat_Level'] == 'High']),
                'medium': len(self.combined_df[self.combined_df['Threat_Level'] == 'Medium']),
                'low': len(self.combined_df[self.combined_df['Threat_Level'] == 'Low'])
            },
            'top_channels': self.combined_df['Channel Name'].value_counts().head(10).to_dict(),
            'iocs': {
                'total_urls': sum(self.combined_df['URLs'].apply(len)),
                'total_ips': sum(self.combined_df['IP_Addresses'].apply(len)),
                'total_hashes': sum(self.combined_df['Hashes'].apply(len))
            }
        }
        
        return analysis
    
    def cluster_messages(self):
        """Cluster similar messages to identify threat campaigns"""
        if self.combined_df is None or len(self.combined_df) == 0:
            return None
            
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.combined_df['Clean_Message'])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(tfidf_matrix)
        
        # Add cluster labels to dataframe
        self.combined_df['Cluster'] = clustering.labels_
        
        return self.combined_df['Cluster'].value_counts().to_dict()
    
    def generate_report(self, output_file='threat_report.json'):
        """Generate a comprehensive threat intelligence report"""
        if self.combined_df is None or len(self.combined_df) == 0:
            return
        
        try:
            # Basic statistics
            stats = {
                'total_messages': len(self.combined_df),
                'unique_channels': self.combined_df['Channel Name'].nunique(),
                'date_range': {
                    'start': self.combined_df['Date'].min().strftime('%Y-%m-%d') if pd.notna(self.combined_df['Date'].min()) else 'Unknown',
                    'end': self.combined_df['Date'].max().strftime('%Y-%m-%d') if pd.notna(self.combined_df['Date'].max()) else 'Unknown'
                },
                'threat_levels': {
                    'high': len(self.combined_df[self.combined_df['Threat_Level'] == 'High']),
                    'medium': len(self.combined_df[self.combined_df['Threat_Level'] == 'Medium']),
                    'low': len(self.combined_df[self.combined_df['Threat_Level'] == 'Low'])
                }
            }
            
            # Top concerning messages
            high_threat_messages = self.combined_df[self.combined_df['Threat_Level'] == 'High']
            top_threats = high_threat_messages.sort_values('Compound_Sentiment', ascending=True).head(5)
            
            threat_details = []
            for _, msg in top_threats.iterrows():
                threat_details.append({
                    'message': msg['Message'][:500],  # Limit message length
                    'channel': msg['Channel Name'],
                    'date': msg['Date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment_score': msg['Compound_Sentiment'],
                    'urls': msg['URLs'],
                    'ips': msg['IP_Addresses'],
                    'hashes': msg['Hashes']
                })
            
            # Channel statistics
            channel_stats = self.combined_df.groupby('Channel Name').agg({
                'Message': 'count',
                'Compound_Sentiment': 'mean',
                'URLs': lambda x: sum(len(i) for i in x),
                'IP_Addresses': lambda x: sum(len(i) for i in x),
                'Hashes': lambda x: sum(len(i) for i in x)
            }).reset_index()
            
            channel_stats = channel_stats.sort_values('Message', ascending=False).head(10)
            channel_details = channel_stats.to_dict('records')
            
            # Compile report
            report = {
                'summary': stats,
                'top_threats': threat_details,
                'channel_statistics': channel_details,
                'ioc_summary': {
                    'total_urls': sum(self.combined_df['URLs'].apply(len)),
                    'total_ips': sum(self.combined_df['IP_Addresses'].apply(len)),
                    'total_hashes': sum(self.combined_df['Hashes'].apply(len))
                },
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save report
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return report
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None
    
    def plot_threat_timeline(self, output_file='threat_timeline.png'):
        """Plot threat levels over time"""
        if self.combined_df is None:
            return False
            
        plt.figure(figsize=(15, 7))
        
        # Group by date and calculate average sentiment
        daily_sentiment = self.combined_df.groupby(
            self.combined_df['Date'].dt.date)['Compound_Sentiment'].mean()
        
        plt.plot(daily_sentiment.index, daily_sentiment.values)
        plt.title('Threat Level Timeline')
        plt.xlabel('Date')
        plt.ylabel('Average Threat Level (Sentiment)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
        
        return True

def main():
    # Initialize processor
    processor = ThreatIntelProcessor()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory")
    
    # Process data
    print("\nLoading data files...")
    if processor.load_data():
        print(f"Found {len(processor.combined_df)} messages to process")
        
        print("\nCleaning and normalizing data...")
        processor.clean_data()
        
        analysis = processor.analyze_threats()
        if analysis:
            print("\n=== Threat Analysis Report ===")
            print(f"Total Messages: {analysis['total_messages']}")
            print(f"Unique Channels: {analysis['unique_channels']}")
            print(f"Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            print("\nThreat Levels:")
            print(f"- High: {analysis['threat_levels']['high']} messages")
            print(f"- Medium: {analysis['threat_levels']['medium']} messages")
            print(f"- Low: {analysis['threat_levels']['low']} messages")
            print("\nIOCs Found:")
            print(f"- URLs: {analysis['iocs']['total_urls']}")
            print(f"- IP Addresses: {analysis['iocs']['total_ips']}")
            print(f"- Hashes: {analysis['iocs']['total_hashes']}")
            
            print("\nTop Active Channels:")
            for channel, count in list(analysis['top_channels'].items())[:5]:
                print(f"- {channel}: {count} messages")
        
        # Cluster messages
        print("\nClustering similar messages...")
        clusters = processor.cluster_messages()
        if clusters:
            print(f"Found {len(clusters)} message clusters")
            print(f"Largest cluster size: {max(clusters.values())} messages")
        
        # Generate report
        print("\nGenerating reports...")
        processor.generate_report(os.path.join('data', 'threat_report.json'))
        print("Threat report saved: data/threat_report.json")
        
        processor.plot_threat_timeline(os.path.join('data', 'threat_timeline.png'))
        print("Timeline visualization saved: data/threat_timeline.png")
        
        print("\nProcessing complete! All files are saved in the data directory.")
    else:
        print("No data files found in the data directory")
        print("Please ensure your Telegram scraping process has completed and saved files to the data directory")

if __name__ == "__main__":
    main()
