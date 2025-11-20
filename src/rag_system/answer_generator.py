"""Answer generation using local LLM models."""

import asyncio
import time
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .models import RetrievalResult, QueryResult, QueryMetrics
from .config import get_config


class AnswerGenerator:
    """Generates answers using local LLM models."""
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._cache = {}  # Simple response cache
        # Simple async lock to prevent concurrent LLM access and tensor conflicts
        self._llm_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the language model asynchronously."""
        # Use async lock for thread safety during initialization
        if self.model is None:
            async with self._llm_lock:
                if self.model is None:  # Check again after acquiring lock
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._load_model)
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm.model_name)
            
            # Fix for PyTorch meta tensor issues
            try:
                # Try loading with specific parameters to avoid meta tensor issues
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.llm.model_name,
                    torch_dtype=torch.float32,  # Always use float32 for compatibility
                    device_map=None,  # Disable automatic device mapping
                    low_cpu_mem_usage=False,  # Disable low memory mode to avoid meta tensors
                    trust_remote_code=False
                )
                
                # Move model to device manually if needed
                if self.config.llm.device != "auto" and self.config.llm.device != "cpu":
                    # Check if CUDA is available before moving to GPU
                    if self.config.llm.device == "cuda" and torch.cuda.is_available():
                        self.model = self.model.to("cuda")
                    else:
                        print("CUDA not available, using CPU")
                        self.model = self.model.to("cpu")
                else:
                    # Ensure model is on CPU
                    self.model = self.model.to("cpu")
                    
            except Exception as e:
                if "accelerate" in str(e) or "meta" in str(e).lower():
                    print("Warning: Model loading issue detected. Trying alternative loading method...")
                    # Alternative loading approach for problematic models
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.config.llm.model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=False
                    )
                    # Force to CPU to avoid device issues
                    self.model = self.model.to("cpu")
                else:
                    raise e
            
            # Create pipeline with thread safety considerations
            # Note: Transformers pipelines should be thread-safe for inference
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.llm.device == "cuda" and torch.cuda.is_available() else -1,
                max_length=self.config.llm.max_length,
                temperature=self.config.llm.temperature,
                do_sample=True if self.config.llm.temperature > 0 else False,
                # Add parameters for better concurrent handling
                batch_size=1,  # Process one at a time for safety
                return_tensors=False  # Return plain text to avoid tensor sharing issues
            )
            
            print(f"✓ LLM model loaded: {self.config.llm.model_name}")
            
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print("Falling back to extractive answer generation...")
            # Fallback to a basic approach
            self.pipeline = None
    
    async def generate_answer(self, query: str, retrieval_results: List[RetrievalResult]) -> QueryResult:
        """
        Generate an answer based on query and retrieved context.
        
        Args:
            query: User query
            retrieval_results: Retrieved relevant documents
            
        Returns:
            QueryResult with answer and metadata
        """
        start_time = time.time()
        
        # Initialize model if needed
        if self.model is None:
            await self.initialize()
        
        # Prepare context from retrieval results
        context_text, used_sources = self._prepare_context(retrieval_results)
        
        # Generate answer
        llm_start = time.time()
        answer = await self._generate_llm_response(query, context_text)
        llm_time = time.time() - llm_start
        
        # Post-process answer
        post_start = time.time()
        final_answer = self._post_process_answer(answer, query)
        post_time = time.time() - post_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        input_tokens = self._count_tokens(f"{query} {context_text}")
        output_tokens = self._count_tokens(final_answer)
        
        metrics = QueryMetrics(
            total_latency=total_time,
            retrieval_time=0.0,  # This would be set by the caller
            llm_time=llm_time,
            post_processing_time=post_time,
            documents_retrieved=len(retrieval_results),
            documents_used_in_answer=len(used_sources),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=0.0  # Free local model
        )
        
        return QueryResult(
            query=query,
            answer=final_answer,
            sources=used_sources,
            metadata={
                'model_name': self.config.llm.model_name,
                'context_length': len(context_text),
                'generation_params': {
                    'max_length': self.config.llm.max_length,
                    'temperature': self.config.llm.temperature
                }
            },
            processing_time=total_time,
            metrics=metrics
        )
    
    def _prepare_context(self, retrieval_results: List[RetrievalResult]) -> tuple[str, List[RetrievalResult]]:
        """
        Prepare context text from retrieval results.
        
        Args:
            retrieval_results: Retrieved results
            
        Returns:
            Tuple of (context_text, sources_used)
        """
        if not retrieval_results:
            return "", []
        
        context_parts = []
        used_sources = []
        max_context_length = self.config.answer_generation.max_context_length
        current_length = 0
        
        # Use configurable number of sources for comprehensive answers
        max_sources = self.config.answer_generation.max_sources
        for i, result in enumerate(retrieval_results[:max_sources]):
            # Clean and prepare content
            content = result.chunk.content.strip()
            
            # Create a detailed context snippet with source info
            source_text = f"Source {i+1} ({result.chunk.metadata.get('document_title', 'Unknown')}): {content}"
            
            # Check if adding this would exceed limit
            if current_length + len(source_text) > max_context_length and context_parts:
                break
            
            context_parts.append(source_text)
            used_sources.append(result)
            current_length += len(source_text)
        
        context_text = "\n\n".join(context_parts)
        return context_text, used_sources
    
    async def _generate_llm_response(self, query: str, context: str) -> str:
        """Generate response using the LLM."""
        if self.pipeline is None:
            # Fallback to simple extraction
            return self._fallback_answer_generation(query, context)
        
        # Check cache first
        cache_key = f"{query}_{hash(context)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Use async lock to prevent concurrent model access and tensor conflicts
        async with self._llm_lock:
            try:
                # Generate response asynchronously
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._generate_sync,
                    prompt
                )
                
                if response and len(response) > 0 and 'generated_text' in response[0]:
                    generated = response[0]['generated_text']
                    
                    # Check if the response is actually the prompt (common with T5 models)
                    if generated.strip() == prompt.strip():
                        # Try to extract just the answer part
                        if "Answer:" in generated:
                            answer_part = generated.split("Answer:")[-1].strip()
                            if answer_part and len(answer_part) > 10:
                                # Cache the result
                                self._cache_response(cache_key, answer_part)
                                return answer_part
                        fallback_answer = self._fallback_answer_generation(query, context)
                        self._cache_response(cache_key, fallback_answer)
                        return fallback_answer
                    
                    # Cache the successful result
                    self._cache_response(cache_key, generated)
                    return generated
                else:
                    fallback_answer = self._fallback_answer_generation(query, context)
                    self._cache_response(cache_key, fallback_answer)
                    return fallback_answer
                    
            except Exception as e:
                print(f"Error generating LLM response: {e}")
                return self._fallback_answer_generation(query, context)
    
    def _generate_sync(self, prompt: str) -> List[Dict[str, str]]:
        """Generate response synchronously with thread safety."""
        # This function is called within the async lock context, so no additional locking needed
        try:
            # Ensure we're working on CPU to avoid tensor device conflicts
            device = self.pipeline.device
            if hasattr(self.pipeline.model, 'device'):
                # Make sure model is on correct device
                if device != self.pipeline.model.device:
                    print(f"Device mismatch detected, ensuring model consistency...")
            
            result = self.pipeline(
                prompt,
                max_new_tokens=self.config.answer_generation.max_new_tokens,
                min_new_tokens=self.config.answer_generation.min_new_tokens,
                temperature=self.config.answer_generation.generation_temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.answer_generation.repetition_penalty,
                # Additional safety parameters
                clean_up_tokenization_spaces=True,
                return_tensors=False
            )
            return result
        except Exception as e:
            # If tensor issues persist, create a more detailed error message
            if "meta" in str(e).lower() or "tensor" in str(e).lower():
                print(f"Tensor device error in concurrent processing: {e}")
                print("Falling back to basic answer generation...")
                raise Exception(f"Model tensor access error during concurrent processing: {e}")
            else:
                raise e
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM."""
        prompt = f"""Based on the provided context, answer the question comprehensively and accurately. Use information from multiple sources when relevant and provide specific details when available.

Context:
{context}

Question: {query}

Answer (provide a detailed response using the context):"""
        
        return prompt
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache a response with size limit."""
        max_cache_size = self.config.answer_generation.max_cache_size
        if len(self._cache) >= max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = response
    
    def _fallback_answer_generation(self, query: str, context: str) -> str:
        """
        Fallback answer generation using simple extraction.
        
        This is used when the LLM is not available.
        """
        if not context.strip():
            return "I don't have enough information to answer this question."
        
        # Extract clean content from sources
        content_lines = []
        for line in context.split('\n'):
            line = line.strip()
            if line and not line.startswith('Source') and len(line) > 20:
                content_lines.append(line)
        
        if not content_lines:
            return "I don't have enough information to answer this question."
        
        # Combine all content
        full_content = ' '.join(content_lines)
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', full_content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        # Find sentences that contain query terms
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            # Also check for keyword matching
            if overlap > 0 or any(word in sentence.lower() for word in query_words):
                scored_sentences.append((sentence, overlap))
        
        if not scored_sentences:
            # If no direct matches, just return the first few sentences from the context
            return '. '.join(sentences[:2]) + '.'
        
        # Sort by overlap and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        max_sentences = self.config.answer_generation.max_fallback_sentences
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        # Create a coherent answer
        answer = '. '.join(top_sentences)
        if not answer.endswith('.'):
            answer += '.'
            
        # If the answer is too short, add more context
        min_length = self.config.answer_generation.min_answer_length
        additional_count = self.config.answer_generation.additional_sentences
        if len(answer) < min_length and len(sentences) > len(top_sentences):
            additional_sentences = [s for s in sentences if s not in top_sentences][:additional_count]
            if additional_sentences:
                answer += ' ' + '. '.join(additional_sentences)
                if not answer.endswith('.'):
                    answer += '.'
            
        return answer
    
    def _post_process_answer(self, answer: str, _query: str) -> str:
        """Post-process the generated answer."""
        if not answer or answer.isspace():
            return "I couldn't generate an answer to your question."
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove prompt text if it leaked through
        prompt_indicators = ["Context:", "Question:", "Answer:", "Based on the following"]
        for indicator in prompt_indicators:
            if indicator in answer:
                parts = answer.split(indicator)
                answer = parts[-1].strip()
        
        # Ensure proper sentence ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: approximate token count
        return len(text.split()) * 1.3  # Rough approximation
    
    async def generate_summary(self, documents: List[str], max_length: int = 200) -> str:
        """
        Generate a summary of multiple documents.
        
        Args:
            documents: List of document texts
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if not documents:
            return ""
        
        # Combine documents
        combined_text = " ".join(documents)
        
        # Limit input length using configurable parameter
        max_input_length = self.config.answer_generation.summary_max_input_length
        if len(combined_text) > max_input_length:
            combined_text = combined_text[:max_input_length] + "..."
        
        prompt = f"Summarize the following text in a few sentences:\n\n{combined_text}\n\nSummary:"
        
        if self.pipeline:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline(prompt, max_length=max_length)
                )
                return response[0]['generated_text'] if response else combined_text[:max_length]
            except:
                pass
        
        # Fallback: extract first few sentences using configurable parameter
        max_sentences = self.config.answer_generation.summary_max_sentences
        sentences = combined_text.split('.')[:max_sentences]
        return '. '.join(sentence.strip() for sentence in sentences if sentence.strip()) + '.'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.config.llm.model_name,
            'model_loaded': self.model is not None,
            'pipeline_loaded': self.pipeline is not None,
            'device': self.config.llm.device,
            'max_length': self.config.llm.max_length,
            'temperature': self.config.llm.temperature
        }