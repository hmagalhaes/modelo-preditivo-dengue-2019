package dengue;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.ui.HtmlSequencePlotting;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;

public class Persistence {

	public final String directory;
	public final String projectName;
	public final String testOutputFile;
	public final String trainingOutputFile;
	public final String networkConfigFile;
	public final String statsFile;

	private Persistence(final String directory, final String projectName) {
		this.projectName = projectName;
		this.directory = setupDirectory(directory, projectName);
		this.testOutputFile = this.directory + "/test-output.csv";
		this.trainingOutputFile = this.directory + "/training-output.csv";
		this.networkConfigFile = this.directory + "/network.json";
		this.statsFile = this.directory + "/stats.txt";
	}

	private String setupDirectory(final String directory, final String projectName) {
		final String fixedDir;
		if (directory.endsWith("/")) {
			fixedDir = directory + projectName;
		} else {
			fixedDir = directory + "/" + projectName;
		}

		final File file = new File(fixedDir);
		if (!file.exists()) {
			file.mkdirs();
		}

		return fixedDir;
	}

	public MultiLayerNetwork restoreNetwork() {
		try {
			return ModelSerializer.restoreMultiLayerNetwork(directory + "/bestModel.bin");
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static Persistence getPersistence(final String directory) {
		final DateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
		final String project = df.format(new Date());

		return new Persistence(directory, project);
	}

	public static Persistence getPersistence(final String directory, final String projectName) {
		return new Persistence(directory, projectName);
	}

	public void save(final MultiLayerConfiguration networkConfig) {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(this.networkConfigFile))) {
			writer.write(networkConfig.toJson());
			writer.write("\n");
		} catch (IOException ex) {
			throw new RuntimeException(ex);
		}
	}

	public void save(final RegressionEvaluation evaluation) {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(this.statsFile))) {
			writer.write(evaluation.stats() + "\n\n");
			writer.write("averageMeanAbsoluteError: " + evaluation.averageMeanAbsoluteError() + "\n");
			writer.write("averageMeanSquaredError: " + evaluation.averageMeanSquaredError() + "\n");
			writer.write("averagerootMeanSquaredError: " + evaluation.averagerootMeanSquaredError() + "\n");
		} catch (IOException ex) {
			throw new RuntimeException(ex);
		}
	}

	private void saveSequencePlotFile(final String title, final String filePath, final List<Double> expectedList,
			final List<Double> obtainedList) {

		final List<List<Writable>> sequence = new ArrayList<>();
		for (int i = 0; i < expectedList.size(); i++) {
			final Writable expected = new DoubleWritable(expectedList.get(i));
			final Writable obtained = new DoubleWritable(obtainedList.get(i));

			final List<Writable> occurrence = new ArrayList<>(2);
			occurrence.add(expected);
			occurrence.add(obtained);

			sequence.add(occurrence);
		}

		final File file = new File(filePath);

		final Schema schema = new SequenceSchema.Builder().addColumnDouble("Esperado").addColumnDouble("Obtido")
				.build();

		try {
			HtmlSequencePlotting.createHtmlSequencePlotFile(title, schema, sequence, file);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void saveTestSequencePlotFile(final List<Double> expectedList, final List<Double> obtainedList) {
		final String title = "Test data: Esperado vs Previsto";
		final String filePath = directory + "/test-plot.html";

		saveSequencePlotFile(title, filePath, expectedList, obtainedList);
	}

	public void saveTrainingSequencePlotFile(final List<Double> expectedList, final List<Double> obtainedList) {
		final String title = "Training data: Esperado vs Previsto";
		final String filePath = directory + "/training-plot.html";

		saveSequencePlotFile(title, filePath, expectedList, obtainedList);
	}

}
