import os
import json
import time
import re
import requests
from typing import List, Dict, Any, Callable, Optional
from urllib.parse import urlparse
import concurrent.futures
from datetime import date

# Third-party imports
from googlesearch import search
import trafilatura

# Updated LangChain imports to use community packages
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import chromadb
# Import configuration
from config import get_config

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI")  # For embeddings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For LLM access

# Configuration - will be overridden based on selected mode
MAX_SEARCH_SUGGESTIONS = 10
RESULTS_PER_SUGGESTION = 5
MAX_WORKERS = 5
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

class ConsultantReportGenerator:
    def __init__(self, progress_callback: Optional[Callable] = None, mode: str = "concise"):
        """Initialize the consultant report generator."""
        # Validate API keys
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Callback for reporting progress
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.progress_callback = progress_callback
        
        # Track search results for UI display
        self.search_results = []
        
        # Get configuration based on mode
        self.config = get_config(mode)
        self.mode = mode.lower()
        
        # Set parameters based on configuration
        self.max_search_suggestions = self.config["search_parameters"]["max_search_suggestions"]
        self.results_per_suggestion = self.config["search_parameters"]["results_per_suggestion"]
        self.max_workers = self.config["search_parameters"]["max_workers"]
        self.report_format = self.config["report_format"]
        
    def update_progress(self, message: str, percent: float):
        """Update progress if callback is provided"""
        if self.progress_callback:
            self.progress_callback(message, percent, self.search_results.copy() if self.search_results else None)
            
    def add_search_result(self, url: str, title: str = "", source: str = ""):
        """Add a search result to the tracked results"""
        self.search_results.append({
            "url": url,
            "title": title or source or self._get_domain(url),
            "source": source or self._get_domain(url)
        })
        
        # Keep only the most recent results (limited to 15 for UI)
        if len(self.search_results) > 15:
            self.search_results = self.search_results[-15:]
            
        # Update progress to refresh the UI
        if self.progress_callback:
            self.progress_callback("Researching content...", 30, self.search_results.copy())
        
    def generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions using LLM."""
        self.update_progress("Generating search suggestions...", 5)
        print(f"Generating search suggestions for: '{query}'")
        
        # OpenRouter API endpoint
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Simplified response format for JSON
        json_schema = {
            "type": "json_object"
        }
        
        payload = {
            "model": "openai/gpt-4o-mini-2024-07-18",
            "messages": [
                {
                    "role": "system",
                    "content": self.config["prompts"]["search_system_prompt"]
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "response_format": json_schema
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"API Error: {result['error']}")
                return self._fallback_suggestions(query)
            structured_output = result["choices"][0]["message"]["content"]
            parsed_json = json.loads(structured_output)
            
            # Return the suggestions array
            if "suggestions" in parsed_json:
                return parsed_json["suggestions"]
            return self._fallback_suggestions(query)
        else:
            print(f"HTTP Error: {response.status_code} - {response.text}")
            return self._fallback_suggestions(query)
    
    def _fallback_suggestions(self, query: str) -> List[str]:
        """Generate fallback suggestions if API fails."""
        if self.mode == "detailed":
            # More balanced, versatile fallback queries for detailed mode
            return [
                f"{query} comprehensive overview",
                f"{query} historical development and evolution",
                f"{query} practical applications and examples",
                f"{query} different perspectives and approaches",
                f"{query} case studies and real-world implementations",
                f"{query} current trends and future directions",
                f"{query} comparative analysis",
                f"{query} best practices and methodologies",
                f"{query} challenges and limitations",
                f"{query} influential works and key contributors"
            ]
        else:
            # Focused fallback queries for concise mode
            return [
                f"{query} overview",
                f"{query} key concepts",
                f"{query} practical examples"
            ]
    
    def search_web(self, suggestions: List[str]) -> Dict[str, List[str]]:
        """Search Google for each suggestion and collect URLs."""
        self.update_progress("Searching for relevant sources...", 15)
        all_results = {}
        
        print("\n--- Searching for relevant web pages ---")
        for i, suggestion in enumerate(suggestions):
            print(f"\nSearching for: '{suggestion}'")
            try:
                # Using googlesearch-python to get search results
                results = list(search(suggestion, num_results=self.results_per_suggestion, lang="en"))
                all_results[suggestion] = results
                
                # Print results for this suggestion
                for j, url in enumerate(results, 1):
                    print(f"  {j}. {url}")
                    # Add to tracked search results for UI
                    self.add_search_result(url, f"Result for: {suggestion}")
                
                # Update progress based on completion of each suggestion
                progress = 15 + (i + 1) / len(suggestions) * 10
                self.update_progress(f"Searching for: {suggestion}", progress)
                
                # Add a small delay to avoid rate limiting
                time.sleep(2)
            except Exception as e:
                print(f"Error searching for '{suggestion}': {e}")
                all_results[suggestion] = []
        
        return all_results
    
    def extract_content(self, url: str) -> str:
        """Extract text content from a URL using trafilatura."""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # Use 'txt' format (not 'text')
                text = trafilatura.extract(downloaded, 
                                          include_links=False, 
                                          include_images=False, 
                                          include_tables=False,
                                          output_format='txt')
                if text:
                    domain = self._get_domain(url)
                    header = f"\n\n{'=' * 80}\nSOURCE: {domain}\nURL: {url}\n{'=' * 80}\n\n"
                    return header + text
            return f"\n\nFailed to extract content from {url}\n\n"
        except Exception as e:
            return f"\n\nError extracting content from {url}: {str(e)}\n\n"
    
    def _get_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def extract_all_content(self, all_results: Dict[str, List[str]]) -> str:
        """Extract content from all URLs using parallel processing."""
        self.update_progress("Extracting content from sources...", 25)
        # Flatten all URLs
        all_urls = []
        for urls in all_results.values():
            all_urls.extend(urls)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in all_urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        print(f"\nDownloading content from {len(unique_urls)} unique URLs...")
        results = []
        
        # Create a progress counter
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start the download tasks
            future_to_url = {executor.submit(self.extract_content, url): url for url in unique_urls}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    if content and len(content) > 100:  # Only include substantial content
                        results.append(content)
                except Exception as e:
                    print(f"\nError processing {url}: {str(e)}")
                
                # Update progress
                completed += 1
                percent_complete = 25 + (completed / len(unique_urls)) * 15
                self.update_progress(f"Downloading content ({completed}/{len(unique_urls)})", percent_complete)
                print(f"Progress: {completed}/{len(unique_urls)} URLs processed", end="\r")
        
        print("\nDownload complete!")
        
        # Combine all content
        combined_content = f"SEARCH QUERY: {self.query}\n\n"
        for i, suggestion in enumerate(all_results.keys(), 1):
            combined_content += f"SUGGESTION {i}: {suggestion}\n"
        combined_content += "\n" + "="*80 + "\n\n"
        combined_content += "\n\n".join(results)
        
        return combined_content
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common HTML artifacts that might have survived
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def split_into_chunks(self, content: str) -> List[Document]:
        """Split content into manageable chunks while preserving semantic meaning."""
        self.update_progress("Processing content...", 45)
        # Extract metadata from the beginning of the file (query and suggestions)
        lines = content.strip().split('\n')
        query = ""
        suggestions = []
        
        for line in lines[:10]:  # Check first 10 lines for metadata
            if line.startswith("SEARCH QUERY:"):
                query = line.replace("SEARCH QUERY:", "").strip()
            elif line.startswith("SUGGESTION"):
                suggestion = line.split(":", 1)[1].strip()
                suggestions.append(suggestion)
        
        # Create a text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        # Extract documents and their sources
        documents = []
        sections = re.split(r'={80}', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Try to extract source information
            source_match = re.search(r'SOURCE: ([^\n]+)', section)
            url_match = re.search(r'URL: ([^\n]+)', section)
            
            source = source_match.group(1) if source_match else "Unknown"
            url = url_match.group(1) if url_match else "Unknown"
            
            # Clean the section text
            cleaned_text = self.clean_text(section)
            
            # Skip if section is too short
            if len(cleaned_text) < 100:
                continue
                
            # Split the section into chunks
            chunks = text_splitter.split_text(cleaned_text)
            
            # Update progress based on section processing
            section_progress = 45 + (i / len(sections)) * 5
            self.update_progress("Processing content sections...", section_progress)
            
            # Create document objects with metadata
            # Join list of suggestions into a string to avoid the Chroma error
            suggestions_str = ", ".join(suggestions) if suggestions else ""
            
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "url": url,
                        "query": query,
                        "related_suggestions": suggestions_str  # Convert list to string
                    }
                )
                documents.append(doc)
        
        print(f"Split content into {len(documents)} chunks")
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a vector store from the documents."""
        self.update_progress("Creating vector embeddings...", 50)
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Create a Chroma vector store in memory (no persistence)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        return vectorstore
    
    def setup_llm_model(self):
        """Configure Gemini 2.0 Flash model via OpenRouter."""
        # Use ChatOpenAI as the wrapper but point to OpenRouter with the Gemini model
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            temperature=0.2,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=4000
        )
        return llm
    
    def create_consultant_outline(self, llm, relevant_docs: List[Document]) -> Dict[str, Any]:
        """Generate a consultant report outline based on the query and relevant documents."""
        self.update_progress("Generating comprehensive report outline...", 60)
        # Combine relevant document content
        contexts = [doc.page_content for doc in relevant_docs[:8]]
        context_text = "\n\n".join(contexts)
        
        # Get today's date as string
        today = date.today()
        today_str = str(today)
        
        # Create outline prompt
        outline_prompt = PromptTemplate(
            input_variables=["query", "context", "today_str"],
            template=self.config["prompts"]["outline_prompt_template"]
        )
        
        # Generate outline
        outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
        response = outline_chain.run(query=self.query, context=context_text, today_str=today_str)
        
        # Parse the JSON response
        try:
            # Clean the response to ensure it's valid JSON
            cleaned_response = re.search(r'\{.*\}', response, re.DOTALL)
            if cleaned_response:
                response = cleaned_response.group(0)
            
            outline = json.loads(response)
            return outline
        except json.JSONDecodeError as e:
            print(f"Error parsing outline JSON: {e}")
            print(f"Raw response: {response}")
            # Fallback simple structure
            return {
                "title": f"A Comprehensive Guide to {self.query}",
                "abstract": "This report explores key aspects, developments, and insights related to this topic, providing valuable context and practical understanding for readers seeking comprehensive knowledge.",
                "sections": [
                    {"section": "Background and Context", "description": "Essential historical and contextual foundation of the topic"},
                    {"section": "Core Concepts and Frameworks", "description": "Key ideas, models, and approaches that structure understanding of this area"},
                    {"section": "Practical Applications", "description": "Real-world implementations and uses in various contexts"},
                    {"section": "Different Perspectives", "description": "Various viewpoints, methodologies, or schools of thought"},
                    {"section": "Challenges and Limitations", "description": "Current obstacles, debates, and boundaries of knowledge"},
                    {"section": "Current Trends", "description": "Recent developments and contemporary practices"},
                    {"section": "Future Directions", "description": "Emerging possibilities and likely future evolution of this field"}
                ]
            }

    def generate_section_content(self, section: Dict[str, str], llm, vectorstore: Chroma, section_index: int, total_sections: int) -> str:
        """Generate content for a specific section of the consultant report."""
        # Calculate progress percentage for this section (spread from 70% to 95%)
        # This spreads the last sections better across the progress bar to prevent the stalling effect
        section_progress_start = 70
        section_progress_range = 25
        section_progress = section_progress_start + (section_index / total_sections) * section_progress_range
        
        # Update progress with more specific information
        self.update_progress(f"Writing section ({section_index+1}/{total_sections}): {section['section']}", section_progress)
        
        # Retrieve documents relevant to this specific section
        section_query = f"{self.query} {section['section']} {section['description']}"
        relevant_docs = vectorstore.similarity_search(section_query, k=8)
        
        # Extract context from relevant documents
        contexts = [doc.page_content for doc in relevant_docs]
        context_text = "\n\n".join(contexts)
        
        # Get today's date as string
        today = date.today()
        today_str = str(today)
        
        # Create prompt for section generation
        section_prompt = PromptTemplate(
            input_variables=["section_name", "section_description", "context", "query", "today_str"],
            template=self.config["prompts"]["section_prompt_template"]
        )
        
        # Provide more granular progress updates for the last sections
        if section_index >= total_sections - 2:
            # For the last two sections, provide more frequent updates
            self.update_progress(f"Finalizing section ({section_index+1}/{total_sections}): {section['section']}...", 
                                section_progress + (section_progress_range / total_sections) * 0.3)
            
        # Generate section content
        section_chain = LLMChain(llm=llm, prompt=section_prompt)
        section_content = section_chain.run(
            section_name=section["section"],
            section_description=section["description"],
            context=context_text,
            query=self.query,
            today_str=today_str
        )
        
        # Clean up any potential headings that the model might have included anyway
        section_content = re.sub(rf"^#+\s*{re.escape(section['section'])}.*?\n", "", section_content, flags=re.IGNORECASE)
        section_content = re.sub(r"^#+.*?\n", "", section_content) # Remove any other headings at the beginning
        
        # Update progress after completing this section
        completed_section_progress = section_progress_start + ((section_index + 1) / total_sections) * section_progress_range
        self.update_progress(f"Completed section ({section_index+1}/{total_sections}): {section['section']}", 
                            completed_section_progress)
        
        return section_content.strip()
    
    def generate_direct_response(self, llm, vectorstore: Chroma) -> str:
        """Generate a direct response for concise mode."""
        self.update_progress("Generating concise response...", 60)
        
        # Get today's date as string
        today = date.today()
        today_str = str(today)
        
        # Retrieve most relevant documents
        relevant_docs = vectorstore.similarity_search(self.query, k=12)
        
        # Extract context from relevant documents
        contexts = [doc.page_content for doc in relevant_docs]
        context_text = "\n\n".join(contexts)
        
        # Create direct response prompt
        response_prompt = PromptTemplate(
            input_variables=["query", "context", "today_str"],
            template=self.config["prompts"]["direct_response_prompt"]
        )
        
        # Generate direct response
        response_chain = LLMChain(llm=llm, prompt=response_prompt)
        
        # Provide progress updates
        self.update_progress("Analyzing information and crafting response...", 75)
        
        # Generate the response
        response = response_chain.run(query=self.query, context=context_text, today_str=today_str)
        
        # Update progress
        self.update_progress("Response complete!", 95)
        
        return response.strip()
    
    def generate_sources(self, documents: List[Document]) -> str:
        """Generate a sources section from the document sources."""
        # Extract unique sources
        sources = {}
        for doc in documents:
            url = doc.metadata.get("url", "")
            source = doc.metadata.get("source", "")
            if url and url != "Unknown" and url not in sources:
                sources[url] = source
        
        # Format sources in a reference format
        references = []
        for i, (url, source) in enumerate(sources.items(), 1):
            references.append(f"{i}. {source}. URL: {url}")
        
        return "\n".join(references)
    
    def generate_consultant_report(self, query: str) -> str:
        """Generate a complete consultant report from the query."""
        self.query = query
        
        # Step 1: Generate search suggestions
        suggestions = self.generate_search_suggestions(query)
        suggestions = suggestions[:self.max_search_suggestions]  # Limit to max suggestions for selected mode
        
        # Step 2: Search the web for each suggestion
        search_results = self.search_web(suggestions)
        
        # Step 3: Extract content from all URLs
        print("\n--- Extracting content from web pages ---")
        content = self.extract_all_content(search_results)
        
        # Save raw content for reference (optional)
        content_file = os.path.join(self.temp_dir, f"{query.replace(' ', '_')}_raw_content.txt")
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Raw content saved to {content_file}")
        
        # Step 4: Process and split content into chunks
        print("\n--- Processing content for report generation ---")
        documents = self.split_into_chunks(content)
        
        # Step 5: Create vector store
        print("Creating vector embeddings (this may take a minute)...")
        vectorstore = self.create_vector_store(documents)
        
        # Step 6: Setup LLM model
        print("Setting up LLM model...")
        llm = self.setup_llm_model()
        
        output_file = ""
        
        # Proceed based on mode
        if self.report_format == "full_report":
            # Generate full detailed report
            print("Generating comprehensive report outline...")
            self.update_progress("Analyzing relevant content...", 55)
            relevant_docs = vectorstore.similarity_search(query, k=10)
            
            # Step 8: Generate consultant report outline
            print("Generating comprehensive report outline...")
            outline = self.create_consultant_outline(llm, relevant_docs)
            print(f"Generated outline for report: {outline['title']}")
            
            # Step 9: Generate content for each section
            print("Generating content section by section...")
            self.update_progress("Writing detailed sections...", 70)
            report_sections = []
            
            # Current date for the report
            current_date = time.strftime("%B %Y")
            
            # Add title page
            report_sections.append(f"# {outline['title']}\n")
            report_sections.append(f"### {current_date}\n")
            
            # Add abstract
            report_sections.append(f"## Abstract\n{outline['abstract']}\n")
            
            # Add table of contents header (actual ToC will be generated by markdown renderer)
            report_sections.append("## Table of Contents\n")
            
            # Generate each section
            total_sections = len(outline['sections'])
            for i, section in enumerate(outline['sections']):
                print(f"Generating section {i+1}/{total_sections}: {section['section']}...")
                section_content = self.generate_section_content(section, llm, vectorstore, i, total_sections)
                report_sections.append(f"## {section['section']}\n{section_content}\n")
                
                # Add a short delay to avoid rate limiting
                time.sleep(1)
            
            # Generate bibliography with sources
            print("Generating bibliography...")
            self.update_progress("Finalizing report...", 95)
            sources = self.generate_sources(documents)
            report_sections.append("## References\n" + sources)
            
            # Add research methodology note
            research_note = """
## Research Methodology

This comprehensive report was generated through a research process that synthesizes information from multiple diverse sources. Content was gathered through advanced search strategies targeting various aspects of the topic, processed using semantic analysis for relevance, and synthesized into a cohesive narrative.

The information presented aims to provide a well-rounded understanding of the subject matter with attention to different perspectives and applications. References to original sources are provided in the bibliography.
"""
            report_sections.append(research_note)
            
            # Combine all sections into the final report
            final_report = "\n\n".join(report_sections)
            
            # Save the consultant report
            output_file = f"{query.replace(' ', '_')}_comprehensive_report.md"
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(final_report)
            
            print(f"\nComprehensive report has been generated and saved to {output_file}")
            self.update_progress("Report complete!", 100)
            
        else:
            # Generate concise direct response
            print("Generating concise direct response...")
            
            # Generate direct response
            response = self.generate_direct_response(llm, vectorstore)
            
            # Add sources for attribution
            sources = self.generate_sources(documents)
            
            # Combine into a simple markdown document
            current_date = time.strftime("%B %Y")
            response_sections = [
                f"# Response: {query}\n### {current_date}\n",
                response,
                "\n\n## Sources\n" + sources
            ]
            
            # Join sections
            final_report = "\n\n".join(response_sections)
            
            # Save the response to file
            output_file = f"{query.replace(' ', '_')}_response.md"
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(final_report)
                
            print(f"\nConcise response has been generated and saved to {output_file}")
            self.update_progress("Response complete!", 100)
        
        return output_file
