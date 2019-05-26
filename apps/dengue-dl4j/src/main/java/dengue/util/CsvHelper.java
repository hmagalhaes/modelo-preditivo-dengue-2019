package dengue.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CsvHelper {
	
	private static final Logger logger = LoggerFactory.getLogger(CsvHelper.class);

	public static String[] readLabels(final String file) {
		try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
			final String line = reader.readLine();
			return line.split(",");
		} catch (IOException ex) {
			throw new RuntimeException(ex);
		}
	}

	public static void writeCsv(final String file, final String[] labels, final double[][] data) {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
			writer.write(String.join(",", labels));
			writer.write("\n");

			for (double[] row : data) {
				final String[] values = toString(row);
				writer.write(String.join(",", values));
				writer.write("\n");
			}
		} catch (IOException ex) {

		}
	}

	private static String[] toString(double[] row) {
		final String[] values = new String[row.length];
		for (int i = 0; i < row.length; i++) {
			values[i] = Double.toString(row[i]);
		}
		return values;
	}

	public static void writeTestResult(List<Double> expected, List<Double> obtained, String outputFile) {
		logger.debug("writing test output => " + outputFile);

		final String[] labels = { "Esperado", "Previsão" };

		final double[][] data = new double[expected.size()][2];
		for (int i = 0; i < expected.size(); i++) {
			data[i][0] = expected.get(i);
			data[i][1] = obtained.get(i);
		}

		writeCsv(outputFile, labels, data);
	}

}
