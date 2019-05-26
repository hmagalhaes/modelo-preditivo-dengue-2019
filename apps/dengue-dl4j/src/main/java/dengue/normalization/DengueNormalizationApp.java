package dengue.normalization;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dengue.util.CsvHelper;

public class DengueNormalizationApp {
	
	private final Logger logger = LoggerFactory.getLogger(getClass());

	private static final String SOURCE = "~/dataset/dataset.csv";
	private static final String TARGET = "~/dataset/dataset-normalized.csv";
	private static final int SKIP_NUM_LINES = 1;
	private static final char DELIMITER = ',';

	public static void main(String... strings) {
		final Columns[] format = { Columns.MES, Columns.TEMP_MAX_MEDIA, Columns.TEMP_MAX_MEDIA,
				Columns.UMIDADE_REL_MEDIA, Columns.UMIDADE_REL_MEDIA, Columns.PRECIPITACAO_ACUMULADA,
				Columns.PRECIPITACAO_ACUMULADA, Columns.VEL_MEDIA_VENTO, Columns.VEL_MEDIA_VENTO, Columns.ATINGIDOS,
				Columns.ATINGIDOS, Columns.ATINGIDOS };
//		final MonthColumns[] format = { MonthColumns.ANO, MonthColumns.INDICE, MonthColumns.MES, MonthColumns.TEMP_MAX_MEDIA,
//		MonthColumns.UMIDADE_REL_MEDIA, MonthColumns.PRECIPITACAO_ACUMULADA, MonthColumns.VEL_MEDIA_VENTO,
//		MonthColumns.ATINGIDOS };
//		final MonthColumns[] format = { MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN,
//				MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN,
//				MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN, MonthColumns.MES_BOOLEAN,
//				MonthColumns.MES_BOOLEAN, MonthColumns.TEMP_MAX_MEDIA, MonthColumns.UMIDADE_REL_MEDIA,
//				MonthColumns.PRECIPITACAO_ACUMULADA, MonthColumns.VEL_MEDIA_VENTO, MonthColumns.ATINGIDOS,
//				MonthColumns.ATINGIDOS, MonthColumns.ATINGIDOS };

		new DengueNormalizationApp().normalizeFile(SOURCE, TARGET, format);
	}

	public void normalizeFile(final String sourceFile, final String targetFile, final Columns[] format) {
		logger.debug("Normalizando => formato => " + Arrays.toString(format));
		
		final double[][] sourceData = readFile(SOURCE);
		System.out.println(Arrays.toString(sourceData));
		final double[][] normalizedData = normalize(sourceData, format);

		final String[] labels = CsvHelper.readLabels(SOURCE);
		CsvHelper.writeCsv(TARGET, labels, normalizedData);
	}

	public double[][] normalize(final double[][] sourceData, final Columns[] format) {
		final int rows = sourceData.length;
		final int cols = sourceData[0].length;
		final double[][] targetData = new double[rows][cols];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				targetData[row][col] = normalize(sourceData, targetData, row, col, format);
			}
		}

		return targetData;
	}

	private double normalize(final double[][] sourceData, final double[][] targetData, final int row, final int col,
			final Columns[] format) {

		final Columns column = format[col];
		final double value = sourceData[row][col];

		return (value - column.min) / (column.max - column.min);
	}

	public double[][] denormalize(final double[][] sourceData, final Columns[] format) {
		final int rows = sourceData.length;
		final int cols = sourceData[0].length;
		final double[][] targetData = new double[rows][cols];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				targetData[row][col] = denormalize(sourceData, targetData, row, col, format);
			}
		}

		return targetData;
	}

	private double denormalize(final double[][] sourceData, final double[][] targetData, final int row, final int col,
			final Columns[] format) {

		final Columns column = format[col];
		final double value = sourceData[row][col];

		return (value * (column.max - column.min)) + column.min;
	}

	private double[][] readFile(final String file) {
		try (CSVRecordReader reader = new CSVRecordReader(SKIP_NUM_LINES, DELIMITER)) {
			reader.initialize(new FileSplit(new File(SOURCE)));

			int columns = 0;
			final List<double[]> dataset = new ArrayList<>();
			while (reader.hasNext()) {
				final List<Writable> writableList = reader.next();
				columns = writableList.size();

				final double[] data = new double[writableList.size()];
				for (int i = 0; i < writableList.size(); i++) {
					final Writable writable = writableList.get(i);
					data[i] = writable.toDouble();
				}

				dataset.add(data);
			}
			logger.debug("Ocorrências Normalizadas => " + dataset.size());
			return dataset.toArray(new double[dataset.size()][columns]);
		} catch (IOException | InterruptedException ex) {
			throw new RuntimeException(ex);
		}
	}

}
