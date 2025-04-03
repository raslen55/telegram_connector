#!/usr/bin/env python3
"""
Telegram to STIX Converter
Converts Telegram threat intelligence data into STIX 2.1 format
"""

import json
import pandas as pd
from datetime import datetime
from stix2 import (
    Identity, Indicator, Relationship, Bundle,
    ThreatActor, Report, Malware, Campaign
)
from stix2.v21 import ObservedData
import ipaddress
import re
from urllib.parse import urlparse

class TelegramStixConverter:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.identity_id = "identity--f431f809-377b-45e0-aa1c-6a4751cae5ff"
        self.identity = Identity(
            id=self.identity_id,
            name="Telegram Threat Intelligence",
            identity_class="organization"
        )
        self.objects = [self.identity]
        self.relationships = []
        
    def load_data(self):
        """Load data from CSV and JSON files"""
        try:
            # Load messages with UTF-8 encoding
            self.df = pd.read_csv(f"{self.data_dir}/complete_dataset.csv", encoding='utf-8')
            
            # Convert date strings to datetime objects
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # Load threat report
            with open(f"{self.data_dir}/threat_report.json", 'r', encoding='utf-8') as f:
                self.threat_report = json.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def extract_iocs(self, row):
        """Extract IOCs from a message"""
        iocs = {
            'urls': row['URLs'].split(',') if isinstance(row['URLs'], str) and row['URLs'] else [],
            'ips': row['IP_Addresses'].split(',') if isinstance(row['IP_Addresses'], str) and row['IP_Addresses'] else [],
            'hashes': row['Hashes'].split(',') if isinstance(row['Hashes'], str) and row['Hashes'] else []
        }
        return iocs
        
    def create_indicator(self, ioc, ioc_type, message_data):
        """Create a STIX Indicator object"""
        if ioc_type == 'url':
            pattern = f"[url:value = '{ioc}']"
            name = f"URL - {urlparse(ioc).netloc}"
        elif ioc_type == 'ipv4':
            pattern = f"[ipv4-addr:value = '{ioc}']"
            name = f"IPv4 - {ioc}"
        elif ioc_type == 'ipv6':
            pattern = f"[ipv6-addr:value = '{ioc}']"
            name = f"IPv6 - {ioc}"
        elif ioc_type == 'hash':
            if len(ioc) == 32:
                pattern = f"[file:hashes.'MD5' = '{ioc}']"
                name = f"MD5 Hash - {ioc[:8]}..."
            elif len(ioc) == 40:
                pattern = f"[file:hashes.'SHA-1' = '{ioc}']"
                name = f"SHA1 Hash - {ioc[:8]}..."
            else:
                pattern = f"[file:hashes.'SHA-256' = '{ioc}']"
                name = f"SHA256 Hash - {ioc[:8]}..."
        
        return Indicator(
            name=name,
            pattern_type="stix",
            pattern=pattern,
            valid_from=message_data['Date'].strftime("%Y-%m-%dT%H:%M:%SZ"),
            labels=["malicious-activity"],
            confidence=self._calculate_confidence(message_data)
        )
    
    def _calculate_confidence(self, message_data):
        """Calculate confidence based on sentiment"""
        compound_sentiment = float(message_data['Compound_Sentiment'])
        if compound_sentiment <= -0.5:  # High threat
            return 85
        elif compound_sentiment <= -0.2:  # Medium threat
            return 65
        else:  # Low threat
            return 45
    
    def create_threat_actor(self, channel_name):
        """Create a STIX Threat Actor object"""
        return ThreatActor(
            name=channel_name,
            sophistication="intermediate",
            resource_level="organization",
            primary_motivation="unknown"
        )
    
    def create_campaign(self, channel_name, messages):
        """Create a STIX Campaign object"""
        first_seen = messages['Date'].min().strftime("%Y-%m-%dT%H:%M:%SZ")
        last_seen = messages['Date'].max().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return Campaign(
            name=f"Campaign - {channel_name}",
            first_seen=first_seen,
            last_seen=last_seen,
            objective="unknown"
        )
    
    def convert(self):
        """Convert Telegram data to STIX format"""
        if not self.load_data():
            return None
            
        try:
            # Process each message
            for _, row in self.df.iterrows():
                # Extract IOCs
                iocs = self.extract_iocs(row)
                message_data = row.to_dict()
                
                # Create indicators for IOCs
                for url in iocs['urls']:
                    if url:
                        indicator = self.create_indicator(url, 'url', message_data)
                        self.objects.append(indicator)
                
                for ip in iocs['ips']:
                    if ip:
                        ip_type = 'ipv6' if ':' in ip else 'ipv4'
                        indicator = self.create_indicator(ip, ip_type, message_data)
                        self.objects.append(indicator)
                
                for hash_value in iocs['hashes']:
                    if hash_value:
                        indicator = self.create_indicator(hash_value, 'hash', message_data)
                        self.objects.append(indicator)
            
            # Create threat actors and campaigns for each channel
            for channel in self.df['Channel Name'].unique():
                channel_messages = self.df[self.df['Channel Name'] == channel]
                
                # Create threat actor
                threat_actor = self.create_threat_actor(channel)
                self.objects.append(threat_actor)
                
                # Create campaign
                campaign = self.create_campaign(channel, channel_messages)
                self.objects.append(campaign)
                
                # Create relationship between threat actor and campaign
                relationship = Relationship(
                    source_ref=threat_actor.id,
                    target_ref=campaign.id,
                    relationship_type="attributed-to"
                )
                self.relationships.append(relationship)
            
            # Create summary report
            report = Report(
                name="Telegram Threat Intelligence Summary",
                published=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                object_refs=[obj.id for obj in self.objects + self.relationships],
                labels=["threat-report"],
                description=f"""
                Threat Intelligence Report
                -------------------------
                Total Messages: {self.threat_report['summary']['total_messages']}
                Unique Channels: {self.threat_report['summary']['unique_channels']}
                
                Threat Levels:
                - High: {self.threat_report['summary']['threat_levels']['high']}
                - Medium: {self.threat_report['summary']['threat_levels']['medium']}
                - Low: {self.threat_report['summary']['threat_levels']['low']}
                
                Date Range:
                - Start: {self.threat_report['summary']['date_range']['start']}
                - End: {self.threat_report['summary']['date_range']['end']}
                
                Top Threats:
                {self._format_top_threats()}
                """
            )
            self.objects.append(report)
            
            # Create STIX bundle
            bundle = Bundle(objects=self.objects + self.relationships)
            
            # Save to file
            with open(f"{self.data_dir}/stix_bundle.json", 'w', encoding='utf-8') as f:
                f.write(bundle.serialize(pretty=True))
            
            print(f"Successfully converted data to STIX format. Output saved to {self.data_dir}/stix_bundle.json")
            return bundle
            
        except Exception as e:
            print(f"Error converting to STIX format: {str(e)}")
            return None
            
    def _format_top_threats(self):
        """Format top threats for the report"""
        if 'top_threats' not in self.threat_report:
            return "No top threats found."
            
        threats = []
        for threat in self.threat_report['top_threats'][:5]:  # Show top 5 threats
            threats.append(f"""
                Channel: {threat['channel']}
                Date: {threat['date']}
                Sentiment Score: {threat['sentiment_score']}
                URLs: {len(threat['urls'])}
                IPs: {len(threat['ips'])}
                Hashes: {len(threat['hashes'])}
                ---""")
        
        return "\n".join(threats)
    
if __name__ == "__main__":
    converter = TelegramStixConverter()
    converter.convert()
