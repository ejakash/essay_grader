import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import org.junit.jupiter.api.Test;
import org.junit.platform.commons.annotation.Testable;

import javax.naming.NamingException;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

class AutograderMainTest {

    @Test
    void getGrammarScore() throws IOException {
        String input = "990384.txt";

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/" + input));
        StringBuilder essay = new StringBuilder();
        String line;
        while ((line = essayReader.readLine()) != null) {
            essay.append(line).append("\n");
        }
        Annotation document = new Annotation(essay.toString());
        pipeline.annotate(document);

        AutograderMain.getGrammarScore(document);
    }

    @Test
    void getSubjects() {
//        String input = "990384.txt";

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//        BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/" + input));
//        StringBuilder essay = new StringBuilder();
//        String line;
//        while ((line = essayReader.readLine()) != null) {
//            essay.append(line).append("\n");
//        }
        String sent = "The food is good.";
        Annotation document = new Annotation(sent);
        pipeline.annotate(document);
        List<CoreMap> sentence = document.get(CoreAnnotations.SentencesAnnotation.class);
        SemanticGraph dependencyParse = sentence.get(0).get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);
        List<CoreLabel> subjects = AutograderMain.getSubjects(dependencyParse);
    }

    @Test
    void getSentenceFormationScore() throws IOException {

        String input = "990384.txt";

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/" + input));
        StringBuilder essay = new StringBuilder();
        String line;
        while ((line = essayReader.readLine()) != null) {
            essay.append(line).append("\n");
        }
//        String sent = "tired, because I travel a lot.";
        Annotation document = new Annotation(essay.toString());
        pipeline.annotate(document);

        AutograderMain.getSentenceFormationScore(document);

    }

    @Test
    void getCoherenceScore() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        String text = "Joan gave a presentation at the conference. Sally helped her with it.";
        Annotation document = new Annotation(text);
        pipeline.annotate(document);
        int coherenceScore = AutograderMain.getCoherenceScore(document);

    }

    @Test
    void getGender() {
        String gender = AutograderMain.getGender("John");
    }
}