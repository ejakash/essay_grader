import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

class BadPosSequence {

    private static List<String> pennTagSet = Arrays.asList("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB");

    static int getBadSequenceCount(CoreMap sentence) {
        List<PosWord> posWords = sentence.get(TokensAnnotation.class).stream().map(PosWord::new).collect(Collectors.toList());
        return (int) badPosSequences.stream().filter(sequence -> Collections.indexOfSubList(posWords, sequence) != -1).count();
    }

    static List<List<PosWord>> getBadSequences(CoreMap sentence) {
        List<PosWord> posWords = sentence.get(TokensAnnotation.class).stream().map(PosWord::new).collect(Collectors.toList());
        return badPosSequences.stream().filter(sequence -> Collections.indexOfSubList(posWords, sequence) != -1).collect(Collectors.toList());
    }

    static List<List<PosWord>> badPosSequences = createPosWords(Arrays.asList(
            "MD,MD",
            "MD,VB,VB",
            "MD,VB,VBD",
            "MD,VB,VBG",
            "MD,VB,VBP",
            "MD,VB,VBZ",
            "MD,VBD",
            "MD,VBG",
            "MD,VBN",
            "MD,VBN,VB",
            "MD,VBN,VBD",
            "MD,VBN,VBN",
            "MD,VBN,VBN",
            "MD,VBN,VBP",
            "MD,VBP",
            "TO,VB,VB",
            "VB,MD",
            "VBD,MD",
            "VBD,VBP",
            "VBN,JJ",
//            "VBN,MD", being experienced would earn you a job.
            "VBP,MD",
            "VBP,VBP",
            "VBZ,VB",
            "WDT,VBN",
            "HAS,HAS",
            "HAS,IS",
            "HAS,MD",
            "HAS,VB",
            "HAS,VBD",
            "HAS,VBG",
            "HAS,VBP",
            "IS,IS",
            "IS,MD",
            "IS,VB",
            "IS,VBD",
            "IS,VBP",
            "MD,HAS",
            "MD,HAS,HAS",
            "MD,HAS,VB",
            "MD,HAS,VBD",
            "MD,HAS,VBN",
            "MD,HAS,VBP",
            "MD,IS",
            "MD,IS,IS",
            "MD,IS,VB",
            "MD,IS,VBD",
            "MD,IS,VBN",
            "MD,IS,VBP",
            "MD,VB,HAS",
            "MD,VBN,HAS",
            "MD,VBN,IS",
            "VB,HAS",
            "VB,IS"
    ));

    static class PosWord {
        String word;
        String pos;
        PosWord(String word, String pos) {
            this.word = word;
            this.pos = pos;
        }

        PosWord(CoreLabel token) {
            this.pos = normalizeCase(token.get(PartOfSpeechAnnotation.class));
            this.word = normalizeCase(token.get(TextAnnotation.class));
        }

        PosWord(String wordOrPos) {
            if(pennTagSet.contains(wordOrPos)) {
                this.pos = normalizeCase(wordOrPos);
                this.word = null;
            }
            else {
                this.pos = null;
                this.word = normalizeCase(wordOrPos);
            }
        }

        private String normalizeCase(String input) {
            return input.toUpperCase();
        }

        @Override
        public boolean equals(Object obj) {
            if(obj instanceof PosWord) {
                PosWord word = (PosWord) obj;
                if(this.word == null) return this.pos.equals(word.pos);
                if(this.pos == null) return this.word.equals(word.word);
                return this.word.equals(word.word) && this.pos.equals(word.pos);
            }
            return super.equals(obj);
        }

        @Override
        public int hashCode() {
            if(word == null) {
                return ("null" + pos).hashCode();
            }
            if(pos == null) {
                return (word + "null").hashCode();
            }
            return ("word" + word + "pos" + pos).hashCode();
        }

        @Override
        public String toString() {
            if(word == null) return pos;
            if(pos == null) return word;
            return word + "|" + pos;
        }
    }

    private static List<List<PosWord>> createPosWords(List<String> sequences) {
        return sequences.stream().map(sequence -> Arrays.stream(sequence.split(",")).map(PosWord::new).collect(Collectors.toList())).collect(Collectors.toList());
    }
}
