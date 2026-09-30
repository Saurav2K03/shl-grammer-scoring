"""
Feature extraction for grammar scoring.
Extracts text-based and audio-based features from transcripts and audio files.
"""
import json
import re
import os
import numpy as np
import pandas as pd
import librosa
import language_tool_python
from collections import Counter


# ============================================================
# TEXT-BASED FEATURES
# ============================================================

def count_syllables(word):
    """Estimate syllable count for a word."""
    word = word.lower()
    if len(word) <= 2:
        return 1
    vowels = 'aeiou'
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def extract_text_features(text, tool=None):
    """Extract grammar and text quality features from a transcript."""
    features = {}
    
    # Handle empty text
    if not text or not text.strip():
        return {k: 0 for k in [
            'word_count', 'unique_word_count', 'type_token_ratio',
            'avg_word_length', 'long_word_ratio', 'sentence_count',
            'avg_sentence_length', 'sentence_length_std', 'max_sentence_length',
            'min_sentence_length', 'short_sentence_ratio', 'long_sentence_ratio',
            'filler_word_count', 'filler_word_ratio', 'repetition_count',
            'repetition_ratio', 'grammar_error_count', 'grammar_error_rate',
            'grammar_error_types', 'spelling_errors', 'grammar_errors_only',
            'punctuation_errors', 'avg_syllables_per_word', 'text_length',
            'comma_count', 'conjunction_count', 'conjunction_ratio',
            'pronoun_count', 'pronoun_ratio', 'article_count',
            'preposition_count', 'complex_word_ratio',
            # v3 new features
            'flesch_kincaid_grade', 'automated_readability_index',
            'errors_per_sentence', 'error_free_sentence_ratio',
            'vocab_sophistication', 'hapax_legomena_ratio',
            'subordinate_clause_count', 'subordinate_clause_ratio',
            'discourse_marker_count', 'discourse_marker_ratio',
            'words_per_minute', 'content_word_ratio',
            'sentence_length_range', 'avg_clause_length',
            'verb_count', 'verb_ratio',
            'modal_verb_count', 'question_count',
            'bigram_overlap_ratio', 'corrected_ttr',
        ]}
    
    text = text.strip()
    
    # --- Word-level features ---
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    features['word_count'] = len(words)
    features['unique_word_count'] = len(set(words))
    features['type_token_ratio'] = features['unique_word_count'] / max(1, features['word_count'])
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / max(1, len(words))
    features['text_length'] = len(text)
    
    # --- Syllable complexity ---
    syllables = [count_syllables(w) for w in words]
    features['avg_syllables_per_word'] = np.mean(syllables) if syllables else 0
    features['complex_word_ratio'] = sum(1 for s in syllables if s >= 3) / max(1, len(words))
    
    # --- Sentence-level features ---
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features['sentence_count'] = len(sentences)
    
    sent_lengths = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
    if sent_lengths:
        features['avg_sentence_length'] = np.mean(sent_lengths)
        features['sentence_length_std'] = np.std(sent_lengths)
        features['max_sentence_length'] = max(sent_lengths)
        features['min_sentence_length'] = min(sent_lengths)
        features['short_sentence_ratio'] = sum(1 for l in sent_lengths if l < 5) / len(sent_lengths)
        features['long_sentence_ratio'] = sum(1 for l in sent_lengths if l > 15) / len(sent_lengths)
    else:
        features['avg_sentence_length'] = 0
        features['sentence_length_std'] = 0
        features['max_sentence_length'] = 0
        features['min_sentence_length'] = 0
        features['short_sentence_ratio'] = 0
        features['long_sentence_ratio'] = 0
    
    # --- Fluency indicators ---
    filler_words = {'um', 'uh', 'hmm', 'like', 'you know', 'i mean', 'basically',
                    'actually', 'literally', 'right', 'so', 'well', 'okay'}
    filler_count = sum(1 for w in words if w in filler_words)
    features['filler_word_count'] = filler_count
    features['filler_word_ratio'] = filler_count / max(1, len(words))
    
    # --- Repetition ---
    if len(words) > 1:
        repetitions = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        features['repetition_count'] = repetitions
        features['repetition_ratio'] = repetitions / max(1, len(words))
    else:
        features['repetition_count'] = 0
        features['repetition_ratio'] = 0
    
    # --- Parts of speech indicators (simple heuristic) ---
    features['comma_count'] = text.count(',')
    
    conjunctions = {'and', 'but', 'or', 'so', 'because', 'although', 'however',
                    'therefore', 'moreover', 'furthermore', 'nevertheless', 'while',
                    'whereas', 'since', 'unless', 'though', 'yet'}
    conj_count = sum(1 for w in words if w in conjunctions)
    features['conjunction_count'] = conj_count
    features['conjunction_ratio'] = conj_count / max(1, len(words))
    
    pronouns = {'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him',
                'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our',
                'they', 'them', 'their', 'theirs'}
    pron_count = sum(1 for w in words if w in pronouns)
    features['pronoun_count'] = pron_count
    features['pronoun_ratio'] = pron_count / max(1, len(words))
    
    articles = {'a', 'an', 'the'}
    features['article_count'] = sum(1 for w in words if w in articles)
    
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
                    'about', 'into', 'through', 'during', 'before', 'after',
                    'above', 'below', 'between', 'under', 'over'}
    features['preposition_count'] = sum(1 for w in words if w in prepositions)
    
    # --- Grammar errors (if tool provided) ---
    if tool:
        try:
            matches = tool.check(text)
            features['grammar_error_count'] = len(matches)
            features['grammar_error_rate'] = len(matches) / max(1, features['word_count'])
            
            error_types = Counter(m.rule_issue_type for m in matches)
            features['grammar_error_types'] = len(error_types)
            features['spelling_errors'] = error_types.get('misspelling', 0)
            features['grammar_errors_only'] = error_types.get('grammar', 0)
            features['punctuation_errors'] = error_types.get('typographical', 0) + error_types.get('punctuation', 0)
        except Exception:
            features['grammar_error_count'] = 0
            features['grammar_error_rate'] = 0
            features['grammar_error_types'] = 0
            features['spelling_errors'] = 0
            features['grammar_errors_only'] = 0
            features['punctuation_errors'] = 0
    else:
        features['grammar_error_count'] = 0
        features['grammar_error_rate'] = 0
        features['grammar_error_types'] = 0
        features['spelling_errors'] = 0
        features['grammar_errors_only'] = 0
        features['punctuation_errors'] = 0
    
    # --- v3: Readability scores ---
    total_syllables = sum(syllables) if syllables else 0
    if features['sentence_count'] > 0 and features['word_count'] > 0:
        asl = features['word_count'] / features['sentence_count']
        asw = total_syllables / features['word_count']
        features['flesch_kincaid_grade'] = 0.39 * asl + 11.8 * asw - 15.59
        chars = sum(len(w) for w in words)
        features['automated_readability_index'] = (
            4.71 * (chars / features['word_count']) +
            0.5 * asl - 21.43
        )
    else:
        features['flesch_kincaid_grade'] = 0
        features['automated_readability_index'] = 0
    
    # --- v3: Error density per sentence ---
    if features['sentence_count'] > 0:
        features['errors_per_sentence'] = features['grammar_error_count'] / features['sentence_count']
    else:
        features['errors_per_sentence'] = 0
    
    # Error-free sentence ratio (approximate: check errors per sentence)
    if tool and features['sentence_count'] > 0:
        try:
            error_free = sum(1 for s in sentences if len(tool.check(s)) == 0)
            features['error_free_sentence_ratio'] = error_free / features['sentence_count']
        except Exception:
            features['error_free_sentence_ratio'] = 0
    else:
        features['error_free_sentence_ratio'] = 0
    
    # --- v3: Vocabulary sophistication ---
    common_words = {'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'could', 'should', 'may', 'might', 'shall',
                    'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this',
                    'that', 'these', 'those', 'and', 'but', 'or', 'so', 'to',
                    'of', 'in', 'on', 'at', 'for', 'with', 'from', 'not', 'no',
                    'if', 'then', 'than', 'very', 'just', 'about', 'up', 'out',
                    'what', 'when', 'where', 'who', 'how', 'all', 'each', 'every'}
    uncommon = [w for w in words if w not in common_words]
    features['vocab_sophistication'] = len(uncommon) / max(1, len(words))
    
    # Hapax legomena ratio (words appearing exactly once)
    word_freq = Counter(words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    features['hapax_legomena_ratio'] = hapax / max(1, len(set(words)))
    
    # Corrected TTR (root TTR) — more stable than raw TTR
    features['corrected_ttr'] = len(set(words)) / max(1, np.sqrt(2 * len(words)))
    
    # --- v3: Clause & structure indicators ---
    subordinators = {'because', 'although', 'though', 'while', 'whereas',
                     'since', 'unless', 'until', 'after', 'before', 'when',
                     'whenever', 'where', 'wherever', 'if', 'whether',
                     'that', 'which', 'who', 'whom', 'whose'}
    sub_count = sum(1 for w in words if w in subordinators)
    features['subordinate_clause_count'] = sub_count
    features['subordinate_clause_ratio'] = sub_count / max(1, len(words))
    
    discourse_markers = {'however', 'therefore', 'moreover', 'furthermore',
                         'nevertheless', 'consequently', 'meanwhile', 'instead',
                         'otherwise', 'indeed', 'certainly', 'specifically',
                         'particularly', 'essentially', 'generally', 'finally',
                         'firstly', 'secondly', 'additionally', 'similarly'}
    dm_count = sum(1 for w in words if w in discourse_markers)
    features['discourse_marker_count'] = dm_count
    features['discourse_marker_ratio'] = dm_count / max(1, len(words))
    
    # Sentence length range (variety)
    if sent_lengths:
        features['sentence_length_range'] = max(sent_lengths) - min(sent_lengths)
    else:
        features['sentence_length_range'] = 0
    
    # Average clause length (approx: words between commas/conjunctions)
    clause_breaks = text.count(',') + text.count(';') + conj_count
    n_clauses = clause_breaks + features['sentence_count']
    features['avg_clause_length'] = features['word_count'] / max(1, n_clauses)
    
    # --- v3: POS-like features ---
    verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'do', 'does', 'did', 'go', 'went',
             'gone', 'come', 'came', 'take', 'took', 'taken', 'make',
             'made', 'get', 'got', 'know', 'knew', 'think', 'thought',
             'see', 'saw', 'want', 'wanted', 'say', 'said', 'tell', 'told',
             'give', 'gave', 'find', 'found', 'try', 'tried', 'feel', 'felt',
             'become', 'became', 'leave', 'left', 'put', 'keep', 'kept',
             'let', 'begin', 'began', 'seem', 'seemed', 'help', 'helped',
             'show', 'showed', 'hear', 'heard', 'play', 'played', 'run',
             'move', 'moved', 'live', 'lived', 'believe', 'believed'}
    v_count = sum(1 for w in words if w in verbs)
    features['verb_count'] = v_count
    features['verb_ratio'] = v_count / max(1, len(words))
    
    modals = {'can', 'could', 'will', 'would', 'shall', 'should', 'may',
              'might', 'must'}
    features['modal_verb_count'] = sum(1 for w in words if w in modals)
    
    # Content word ratio (non-function words)
    function_words = common_words | {'not', 'no', 'just', 'very', 'too', 'also'}
    content_count = sum(1 for w in words if w not in function_words)
    features['content_word_ratio'] = content_count / max(1, len(words))
    
    # Question count
    features['question_count'] = text.count('?')
    
    # --- v3: Words per minute (needs duration from audio, approximate from text) ---
    features['words_per_minute'] = 0  # Will be overridden if duration available
    
    # --- v3: Bigram overlap (lexical coherence) ---
    if len(words) >= 3:
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        features['bigram_overlap_ratio'] = 1 - (unique_bigrams / max(1, len(bigrams)))
    else:
        features['bigram_overlap_ratio'] = 0
    
    return features


# ============================================================
# AUDIO-BASED FEATURES
# ============================================================

def extract_audio_features(audio_path, transcript_segments=None):
    """Extract audio quality features from a .wav file."""
    features = {}
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        features['duration'] = duration
        
        # --- Energy features ---
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        
        # --- Silence/pause detection ---
        # Frames below threshold are considered silent
        threshold = np.mean(rms) * 0.3
        silent_frames = np.sum(rms < threshold)
        total_frames = len(rms)
        features['silence_ratio'] = silent_frames / max(1, total_frames)
        
        # --- Speaking rate (from segments if available) ---
        if transcript_segments and len(transcript_segments) > 0:
            total_words = sum(len(seg['text'].split()) for seg in transcript_segments)
            speech_duration = transcript_segments[-1]['end'] - transcript_segments[0]['start']
            features['speaking_rate'] = total_words / max(0.1, speech_duration) * 60  # words per min
            
            # Pause analysis
            pauses = []
            for i in range(1, len(transcript_segments)):
                gap = transcript_segments[i]['start'] - transcript_segments[i-1]['end']
                if gap > 0.3:  # pauses > 300ms
                    pauses.append(gap)
            features['pause_count'] = len(pauses)
            features['total_pause_duration'] = sum(pauses) if pauses else 0
            features['avg_pause_duration'] = np.mean(pauses) if pauses else 0
            features['max_pause_duration'] = max(pauses) if pauses else 0
        else:
            features['speaking_rate'] = 0
            features['pause_count'] = 0
            features['total_pause_duration'] = 0
            features['avg_pause_duration'] = 0
            features['max_pause_duration'] = 0
        
        # --- Spectral features ---
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # --- MFCC features (voice quality) ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # --- Zero crossing rate (voice quality indicator) ---
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # --- Pitch features ---
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            pitch_values = pitch_values[pitch_values > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
        except Exception:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            
    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {e}")
        # Return zeros for all features
        feature_names = ['duration', 'rms_mean', 'rms_std', 'rms_max', 'silence_ratio',
                        'speaking_rate', 'pause_count', 'total_pause_duration',
                        'avg_pause_duration', 'max_pause_duration',
                        'spectral_centroid_mean', 'spectral_centroid_std',
                        'zcr_mean', 'zcr_std', 'pitch_mean', 'pitch_std', 'pitch_range']
        feature_names += [f'mfcc_{i}_mean' for i in range(13)]
        feature_names += [f'mfcc_{i}_std' for i in range(13)]
        features = {k: 0 for k in feature_names}
    
    return features


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_all_features(df, transcripts, audio_dir, use_grammar_tool=True):
    """Extract all features for a dataframe of audio files.
    
    Args:
        df: DataFrame with 'filename' column
        transcripts: dict of {filename: {"text": ..., "segments": ...}}
        audio_dir: path to audio files
        use_grammar_tool: whether to use language_tool for grammar checking
    
    Returns:
        DataFrame with all features
    """
    if use_grammar_tool:
        print("Initializing LanguageTool (this may take a moment)...")
        tool = language_tool_python.LanguageTool('en-US')
    else:
        tool = None
    print("Done.")
    
    all_features = []
    total = len(df)
    for i, row in df.iterrows():
        filename = row['filename']
        transcript = transcripts.get(filename, {"text": "", "segments": []})
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        audio_path = os.path.join(audio_dir, filename)
        
        # Extract text features
        text_feats = extract_text_features(text, tool=tool)
        
        # Extract audio features
        audio_feats = extract_audio_features(audio_path, segments)
        
        # Combine
        combined = {**text_feats, **audio_feats}
        
        # Compute words per minute using audio duration
        if 'duration' in audio_feats and audio_feats['duration'] > 0:
            combined['words_per_minute'] = text_feats.get('word_count', 0) / (audio_feats['duration'] / 60.0)
        
        # --- v3: Whisper confidence features from segment metadata ---
        if segments:
            avg_logprobs = [s.get('avg_logprob', 0) for s in segments if 'avg_logprob' in s]
            no_speech_probs = [s.get('no_speech_prob', 0) for s in segments if 'no_speech_prob' in s]
            compression_ratios = [s.get('compression_ratio', 0) for s in segments if 'compression_ratio' in s]
            temperatures = [s.get('temperature', 0) for s in segments if 'temperature' in s]
            
            if avg_logprobs:
                combined['whisper_avg_logprob_mean'] = np.mean(avg_logprobs)
                combined['whisper_avg_logprob_min'] = np.min(avg_logprobs)
                combined['whisper_avg_logprob_std'] = np.std(avg_logprobs)
            else:
                combined['whisper_avg_logprob_mean'] = 0
                combined['whisper_avg_logprob_min'] = 0
                combined['whisper_avg_logprob_std'] = 0
            
            if no_speech_probs:
                combined['whisper_no_speech_mean'] = np.mean(no_speech_probs)
                combined['whisper_no_speech_max'] = np.max(no_speech_probs)
                combined['whisper_high_no_speech_ratio'] = sum(1 for p in no_speech_probs if p > 0.5) / len(no_speech_probs)
            else:
                combined['whisper_no_speech_mean'] = 0
                combined['whisper_no_speech_max'] = 0
                combined['whisper_high_no_speech_ratio'] = 0
            
            if compression_ratios:
                combined['whisper_compression_mean'] = np.mean(compression_ratios)
                combined['whisper_compression_max'] = np.max(compression_ratios)
                combined['whisper_compression_std'] = np.std(compression_ratios)
            else:
                combined['whisper_compression_mean'] = 0
                combined['whisper_compression_max'] = 0
                combined['whisper_compression_std'] = 0
            
            if temperatures:
                combined['whisper_temp_mean'] = np.mean(temperatures)
                combined['whisper_temp_max'] = np.max(temperatures)
                combined['whisper_high_temp_ratio'] = sum(1 for t in temperatures if t > 0) / len(temperatures)
            else:
                combined['whisper_temp_mean'] = 0
                combined['whisper_temp_max'] = 0
                combined['whisper_high_temp_ratio'] = 0
        else:
            # No segments available
            for key in ['whisper_avg_logprob_mean', 'whisper_avg_logprob_min', 'whisper_avg_logprob_std',
                       'whisper_no_speech_mean', 'whisper_no_speech_max', 'whisper_high_no_speech_ratio',
                       'whisper_compression_mean', 'whisper_compression_max', 'whisper_compression_std',
                       'whisper_temp_mean', 'whisper_temp_max', 'whisper_high_temp_ratio']:
                combined[key] = 0
        
        combined['filename'] = filename
        all_features.append(combined)
        
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] features extracted")
    
    if tool:
        tool.close()
    
    return pd.DataFrame(all_features)
