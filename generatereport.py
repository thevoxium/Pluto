import os
import json
import time
import re
import requests
from typing import List, Dict, Any, Callable, Optional
from urllib.parse import urlparse
import concurrent.futures

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

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI")  # For embeddings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For LLM access

# Configuration
MAX_SEARCH_SUGGESTIONS = 10
RESULTS_PER_SUGGESTION = 5
MAX_WORKERS = 5
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

class ConsultantReportGenerator:
    def __init__(self, progress_callback: Optional[Callable] = None):
        """Initialize the consultant report generator."""
        # Validate API keys
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Callback for reporting progress
        self.progress_callback = progress_callback
        
        # Track search results for UI display
        self.search_results = []
        
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
                    "content": """You are a versatile research assistant with expertise in diverse fields including humanities, sciences, arts, technology, business, and more.

Your task is to analyze a query and generate 10 diverse search terms that will gather comprehensive information across multiple dimensions of the topic.

Adapt your search strategy to the nature of the query:
1. For scientific/technical topics: Include search terms covering theoretical foundations, practical applications, and recent developments
2. For humanities/social sciences: Include terms exploring historical context, different perspectives, social implications, and case studies
3. For practical subjects: Include terms for methodologies, best practices, examples, and comparative analyses
4. For creative fields: Include terms for techniques, influential works, evolution of the field, and contemporary trends

Your search terms should balance:
- Foundational knowledge (30%)
- Intermediate insights (40%)
- Advanced perspectives (30%)

Deliberately include search terms that will uncover:
- Different viewpoints and interpretations
- Historical evolution and context
- Contemporary applications and relevance
- Future trends and emerging directions
- Practical examples and case studies
- Comparative analyses
- Cultural or geographic variations when relevant

Format your response as a JSON object with a "suggestions" array containing exactly 10 search queries."""
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
        # More balanced, versatile fallback queries
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
    
    def search_web(self, suggestions: List[str]) -> Dict[str, List[str]]:
        """Search Google for each suggestion and collect URLs."""
        self.update_progress("Searching for relevant sources...", 15)
        all_results = {}
        
        print("\n--- Searching for relevant web pages ---")
        for i, suggestion in enumerate(suggestions):
            print(f"\nSearching for: '{suggestion}'")
            try:
                # Using googlesearch-python to get search results
                results = list(search(suggestion, num_results=RESULTS_PER_SUGGESTION, lang="en"))
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
        
        # Create outline prompt for a versatile approach across fields
        outline_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are a versatile research assistant with expertise in diverse fields. Your goal is to create a comprehensive, 
            insightful deep-dive report on any topic, adapting your approach to the subject matter.
            
            Based on the following query and retrieved information, create a detailed outline for an in-depth report. 
            The outline should be appropriate to the field and nature of the query - whether it's humanities, sciences, 
            arts, business, technology, or any other domain.
            
            Your outline must include:
            
            1. An engaging, informative title that captures the essence of the topic
            2. A concise abstract (150-200 words) that previews the main insights and value of the report
            3. 5-7 major section headers that:
               - Logically progress through the topic
               - Adapt naturally to the subject matter
               - Cover diverse aspects appropriate to the field
               - Include contextual background where helpful
               - Examine practical applications when relevant
               - Consider different perspectives or approaches
               - Address current developments and future directions
               
            4. For each section, provide a brief description (50-75 words) of what will be covered
            
            IMPORTANT GUIDELINES:
            - Adapt your structure to the nature of the query - different types of questions require different approaches
            - For scientific/technical topics: balance conceptual understanding with practical implications
            - For humanities/social topics: include historical context, different perspectives, and real-world relevance
            - For practical queries: focus on methodologies, examples, and applications
            - For creative topics: explore techniques, notable works, and evolutionary trends
            - Avoid overly abstract or purely theoretical sections unless the query specifically calls for them
            - Ensure the outline would be helpful to someone seeking comprehensive understanding
            
            QUERY: {query}
            
            RELEVANT INFORMATION:
            {context}
            
            Format your response as JSON with the following structure:
            {{
                "title": "The engaging title",
                "abstract": "Concise abstract previewing main insights",
                "sections": [
                    {{"section": "Section Name", "description": "Brief description of what this section will cover"}},
                    // other sections here
                ]
            }}
            """
        )
        
        # Generate outline
        outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
        response = outline_chain.run(query=self.query, context=context_text)
        
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
        
        # Create prompt for section generation with adaptive approach
        section_prompt = PromptTemplate(
            input_variables=["section_name", "section_description", "context", "query"],
            template="""
            You are a versatile research assistant writing a section of a comprehensive report on "{query}".
            
            SECTION: {section_name}
            DESCRIPTION: {section_description}
            
            Relevant information from research:
            {context}
            
            Write a comprehensive, insightful section (approximately 800-1200 words) that provides valuable understanding 
            to readers. Your writing should:
            
            1. ADAPT TO THE SUBJECT MATTER:
               - For scientific/technical topics: Explain concepts clearly with appropriate depth
               - For humanities/social topics: Provide context, varied perspectives, and meaningful examples
               - For practical topics: Include methodologies, applications, and real-world examples
               - For creative/artistic topics: Explore techniques, influences, and significant works
            
            2. BE INFORMATIVE AND ENGAGING:
               - Use a clear, accessible writing style
               - Explain specialized terminology when needed
               - Include specific examples, cases, or illustrations
               - Weave in insightful analysis and not just facts
               - Present information in a logical flow
            
            3. OFFER BALANCED COVERAGE:
               - Include diverse viewpoints when relevant
               - Consider historical context when helpful
               - Connect theory to practice where appropriate
               - Address nuances and complexities
               - Acknowledge ongoing debates or uncertainties
            
            4. PROVIDE DEPTH AND SUBSTANCE:
               - Go beyond surface-level information
               - Include specific data, figures, or examples when relevant
               - Discuss implications and significance
               - Connect this section to the broader topic
               - Offer insights that would help readers develop a sophisticated understanding
            
            This section should be informative, well-structured, and directly relevant to helping readers develop a 
            comprehensive understanding of the topic. Use a conversational yet substantive tone.
            
            Do not include the section title in your response, as it will be added separately.
            """
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
            query=self.query
        )
        
        # Clean up any potential headings that the model might have included anyway
        section_content = re.sub(rf"^#+\s*{re.escape(section['section'])}.*?\n", "", section_content, flags=re.IGNORECASE)
        section_content = re.sub(r"^#+.*?\n", "", section_content) # Remove any other headings at the beginning
        
        # Update progress after completing this section
        completed_section_progress = section_progress_start + ((section_index + 1) / total_sections) * section_progress_range
        self.update_progress(f"Completed section ({section_index+1}/{total_sections}): {section['section']}", 
                            completed_section_progress)
        
        return section_content.strip()
    
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
        
        # Step 7: Retrieve relevant documents for the main query
        print("Retrieving most relevant content for query...")
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
        return output_file
