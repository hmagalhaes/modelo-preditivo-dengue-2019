package dengue.network;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dengue.Persistence;
import dengue.util.CsvHelper;

public class DengueNetworkApp {

	/**
	 * Mensurações da performance da rede são feitas em cima do mini batch, invés de
	 * a cada amostra.<br/>
	 * <br/>
	 * Trabalhar em batches aumenta a performance de treinamento, porém exige que os
	 * batches sejam bem distribuídos quanto às classes de saída (no caso de
	 * classificadores). Caso contrário pode criar um viés na rede, já que a função
	 * de perda, que embasa a retropropagação, é feita em cima do batch.
	 */
	private static final int BATCH_SIZE = 1;
	private static final int SKIP_NUM_LINES = 1;
	private static final char DELIMITER = ',';
	private static final int EPOCHS_NUMBER = 5000;

	private static final String PROJECTS_DIR = "./modelos/MODELO-7";
	private static final String TRAIN_DATASET_SORTED = PROJECTS_DIR + "/input/dataset-normalized.sorted.csv";
	private static final String TRAIN_DATASET_RANDOM = PROJECTS_DIR + "/input/dataset-normalized.random.csv";
	private static final String TEST_DATASET = PROJECTS_DIR + "/input/testset-normalized.csv";
	private static final boolean RANDOM_DATASET = false;

	private static final long SEED = 2545856984l;
	private static final double LEARNING_RATE = 0.003;
	private static final double MOMENTUM = 0.9;
	private static final int INPUT_NODES = 18;
	private static final int LABEL_COL_INDEX = INPUT_NODES;
	private static final int LAYER_1_NODES = 6;
	private static final int OUPUT_NODES = 1;
	private static final LossFunction MEAN_SQUARE_ERROR_FUNCTION = LossFunction.MSE;
	private static final boolean ENABLE_ONLINE_STATS = false;
	
	private final Logger logger = LoggerFactory.getLogger(getClass());

	public static void main(String[] args) throws Exception {
		new DengueNetworkApp().run();
	}

	public void run() throws Exception {
		try {
			final String trainingDataset = RANDOM_DATASET ? TRAIN_DATASET_RANDOM : TRAIN_DATASET_SORTED;
			final DataSetIterator dataSetIterator = getDataIterator(trainingDataset);
			final DataSetIterator testSetIterator = getDataIterator(TEST_DATASET);

//			final TrainingStrategy strategy = new SerializedStrategy();
//			final TrainingStrategy strategy = new StraightStrategy();
			final TrainingStrategy strategy = new EarlyStopStrategy();

			strategy.run(dataSetIterator, testSetIterator);
		} finally {
			try {
				UIServer.stopInstance();
			} catch (RuntimeException ex) {
			}
		}
	}

	private interface TrainingStrategy {

		void run(final DataSetIterator dataSetIterator, final DataSetIterator testSetIterator) throws Exception;
	}

	private class SerializedStrategy implements TrainingStrategy {

		public void run(final DataSetIterator dataSetIterator, final DataSetIterator testSetIterator) throws Exception {

			final TrainingListener statsListener = createStatsListener();
			final Persistence persistence = Persistence.getPersistence(PROJECTS_DIR, "/early-stop");

			final MultiLayerNetwork network = persistence.restoreNetwork();
			network.setListeners(statsListener);

			// Testing

			logger.info("----------");

			evaluate(network, dataSetIterator, testSetIterator, persistence);

			logger.info("----------");

			test(network, dataSetIterator, testSetIterator, persistence);
		}
	}

	private class StraightStrategy implements TrainingStrategy {

		public void run(final DataSetIterator dataSetIterator, final DataSetIterator testSetIterator) throws Exception {

			final TrainingListener statsListener = createStatsListener();
			final Persistence persistence = Persistence.getPersistence(PROJECTS_DIR);

			final MultiLayerConfiguration config = createModelConfig();
			persistence.save(config);

			final MultiLayerNetwork network = new MultiLayerNetwork(config);
			network.setListeners(statsListener);
			network.init();

			// Training

			network.fit(dataSetIterator, EPOCHS_NUMBER);

			// Testing

			logger.info("----------");

			evaluate(network, dataSetIterator, testSetIterator, persistence);

			logger.info("----------");

			test(network, dataSetIterator, testSetIterator, persistence);
		}
	}

	private class EarlyStopStrategy implements TrainingStrategy {

		public void run(final DataSetIterator dataSetIterator, final DataSetIterator testSetIterator) throws Exception {

			final Persistence persistence = Persistence.getPersistence(PROJECTS_DIR);
			final boolean average = true;
			final EarlyStoppingModelSaver modelSaver = new LocalFileModelSaver(persistence.directory);
			final MultiLayerConfiguration networkConfig = createModelConfig();

			final EarlyStoppingConfiguration earlyConfig = new EarlyStoppingConfiguration.Builder()
					.epochTerminationConditions(new MaxEpochsTerminationCondition(EPOCHS_NUMBER))
//				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
					.scoreCalculator(new DataSetLossCalculator(testSetIterator, average)).evaluateEveryNEpochs(1)
					.modelSaver(modelSaver).build();

			persistence.save(networkConfig);

			final EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlyConfig, networkConfig, dataSetIterator);

			// training

			final EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
			logger.info("Termination reason: " + result.getTerminationReason());
			logger.info("Termination details: " + result.getTerminationDetails());
			logger.info("Total epochs: " + result.getTotalEpochs());
			logger.info("Best epoch number: " + result.getBestModelEpoch());
			logger.info("Score at best epoch: " + result.getBestModelScore());

			final TrainingListener statsListener = createStatsListener();

			final MultiLayerNetwork network = result.getBestModel();
			network.setListeners(statsListener);

			// testing

			logger.info("----------");

			evaluate(network, dataSetIterator, testSetIterator, persistence);

			logger.info("----------");

			test(network, dataSetIterator, testSetIterator, persistence);
		}
	}

	private void evaluate(final MultiLayerNetwork network, final DataSetIterator dataSetIterator,
			final DataSetIterator testSetIterator, final Persistence persistence) {

		testSetIterator.reset();

		final RegressionEvaluation evaluation = network.evaluateRegression(testSetIterator);

		persistence.save(evaluation);

		logger.info(evaluation.stats());
		logger.info("lastScore: " + network.score());
		logger.info("averageMeanAbsoluteError: " + evaluation.averageMeanAbsoluteError());
		logger.info("averageMeanSquaredError: " + evaluation.averageMeanSquaredError());
		logger.info("averagerootMeanSquaredError: " + evaluation.averagerootMeanSquaredError());
	}

	private TrainingListener createStatsListener() {
		if (!ENABLE_ONLINE_STATS) {
			final int printingFrequency = 1;
			return new ScoreIterationListener(printingFrequency);
		}

		// Configure where the network information is to be stored.
		// Alternative: new FileStatsStorage(File), for saving and loading later
		final StatsStorage statsStorage = new InMemoryStatsStorage();

		final UIServer uiServer = UIServer.getInstance();
		uiServer.attach(statsStorage);

		return new StatsListener(statsStorage);
	}

	private void test(final MultiLayerNetwork network, final DataSetIterator dataSetIterator,
			final DataSetIterator testSetIterator, final Persistence persistence) {

		final boolean testSet = true;
		test(network, dataSetIterator, persistence, !testSet);
		test(network, testSetIterator, persistence, testSet);
	}

	private List<Double> test(final MultiLayerNetwork network, final DataSetIterator iterator,
			final Persistence persistence, final boolean testSet) {

		final List<Double> expectedList = new ArrayList<>();
		final List<Double> obtainedList = new ArrayList<>();

		iterator.reset();
		while (iterator.hasNext()) {
			final DataSet dataSet = iterator.next();
			final INDArray features = dataSet.getFeatures();

			final INDArray output = network.output(features);
			final double[] expected = dataSet.getLabels().toDoubleVector();
			final double[] obtained = output.toDoubleVector();

			add(expected, expectedList);
			add(obtained, obtainedList);
		}
		iterator.reset();

		logger.info("Output: \n  obtained: " + obtainedList + "\n  expected: " + expectedList);

		if (testSet) {
			CsvHelper.writeTestResult(expectedList, obtainedList, persistence.testOutputFile);
			persistence.saveTestSequencePlotFile(expectedList, obtainedList);
		} else {
			CsvHelper.writeTestResult(expectedList, obtainedList, persistence.trainingOutputFile);
			persistence.saveTrainingSequencePlotFile(expectedList, obtainedList);
		}

		return obtainedList;
	}

	private void add(final double[] source, final List<Double> target) {
		for (double value : source) {
			target.add(value);
		}
	}

	private DataSetIterator getDataIterator(final String file) throws Exception {
		final CSVRecordReader recordReader = new CSVRecordReader(SKIP_NUM_LINES, DELIMITER);
		recordReader.initialize(new FileSplit(new File(file)));
		logger.debug("loading iterator: " + file);
		return new RecordReaderDataSetIterator.Builder(recordReader, BATCH_SIZE).regression(LABEL_COL_INDEX).build();
	}

	private MultiLayerConfiguration createModelConfig() {
		final Layer layer0 = new DenseLayer.Builder().nIn(INPUT_NODES).nOut(LAYER_1_NODES).weightInit(WeightInit.XAVIER)
				.activation(Activation.RELU).hasBias(true).biasInit(1).build();
		final Layer layer1 = new OutputLayer.Builder().nIn(LAYER_1_NODES).nOut(OUPUT_NODES)
				.weightInit(WeightInit.XAVIER).lossFunction(MEAN_SQUARE_ERROR_FUNCTION).activation(Activation.IDENTITY)
				.hasBias(false).build();

		final IUpdater updater = new Nesterovs(LEARNING_RATE, MOMENTUM);
//		final IUpdater updater = new Sgd(LEARNING_RATE);

		return new NeuralNetConfiguration.Builder().seed(SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(updater).list()
				.layer(0, layer0).layer(1, layer1).backpropType(BackpropType.Standard).build();
	}

}
