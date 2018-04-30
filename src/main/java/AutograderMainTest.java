import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.junit.jupiter.api.Test;
import org.junit.platform.commons.annotation.Testable;

import javax.naming.NamingException;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
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

        AutograderMain.getSentenceFormationScore(document,"high");

    }
}