import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class PIDD {
    static final int batchSize = 128;
    static final int outputSize = 2;
    static final int classIndex = 8;
    static final double learningRate = 0.001;
    static final int inputSize = 8;
    static final int numHiddenNodes = 15;
    static int maxIterations = 10000;
    private final String mode;

    static final String dataSetName = "diabetes";

    private MultiLayerNetwork model;

    public PIDD(String mode) {
        this.mode = mode;
        try {
            if (this.mode.equals("load")) {
                System.out.println("Loading The model");
                this.model = ModelSerializer.restoreMultiLayerNetwork(new File("model.zip"));
            } else if (this.mode.equals("build")) {
                this.buildModel();
            }
        } catch (IOException ignored) {
            this.buildModel();
        }

    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        PIDD pidd = new PIDD("load");

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(inMemoryStatsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        pidd.getModel().setListeners(new StatsListener(inMemoryStatsStorage));

        pidd.trainModelAndEvaluate();

    }

    public void buildModel() {
        System.out.println("Building the model...");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(98)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(learningRate))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .list()
                .layer(
                        new DenseLayer.Builder()
                                .nIn(inputSize)
                                .nOut(numHiddenNodes)
                                .build())
                .layer(
                        new DenseLayer.Builder()
                                .nOut(numHiddenNodes)
                                .build())
                .layer(
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputSize)
                                .activation(Activation.SOFTMAX)
                                .build())
                .build();
        this.model = new MultiLayerNetwork(configuration);
        this.model.init();
    }

    public DataSetIterator getDataSetIterator(String mode) throws IOException, InterruptedException {
        String filename;
        if (mode.equals("train") || mode.equals("test")) {
            filename = dataSetName + "-" + mode + ".csv";
        } else return null;
//        System.out.println(mode + "," + filename);
        File testFile = new ClassPathResource(filename).getFile();
        RecordReader testRecordReader = new CSVRecordReader();
        testRecordReader.initialize(new FileSplit(testFile));
        return new RecordReaderDataSetIterator(testRecordReader, batchSize,
                classIndex, outputSize);
    }

    public void trainModelAndEvaluate() throws IOException, InterruptedException {

        DataSetIterator testSetIterator = this.getDataSetIterator("test");
        DataSetIterator trainDataSetIterator = this.getDataSetIterator("train");

        Evaluation evaluation;
        double startingAccuracy = 0;
        if (this.mode.equals("load")) {
            evaluation = model.evaluate(testSetIterator);
            startingAccuracy = evaluation.accuracy();
            System.out.println("Starting accuracy " + startingAccuracy);
        }
        int i = 1;
        while (startingAccuracy < 0.8 && i < maxIterations) {
            testSetIterator.reset();
            trainDataSetIterator.reset();


            System.out.println("Epoch " + (i++));
            model.fit(trainDataSetIterator);
            evaluation = model.evaluate(testSetIterator);

            if (evaluation.accuracy() > startingAccuracy) {
                startingAccuracy = evaluation.accuracy();
                System.out.println("new accuracy " + evaluation.accuracy());
                System.out.println(evaluation.stats());
                System.out.println("Saving model !");
                ModelSerializer.writeModel(model, new File("model.zip"), true);
            }

            if (i % 20 == 0) {
                System.out.println("Current accuracy " + evaluation.accuracy());
            }
            if (i % 60 == 0) {
                System.out.println("Starting accuracy " + startingAccuracy);
                System.out.println("Current accuracy " + evaluation.accuracy());
                System.out.println(evaluation.stats());
            }
        }
    }
}
