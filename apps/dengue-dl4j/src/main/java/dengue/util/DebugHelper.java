package dengue.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DebugHelper {

	private final Logger logger = LoggerFactory.getLogger(getClass());

	public void printDataSet(final String name, final DataSetIterator iterator) {
		logger.debug("------- " + name);
		iterator.reset();
		while (iterator.hasNext()) {
			final DataSet dataSet = iterator.next();
			final INDArray features = dataSet.getFeatures();
			final INDArray labels = dataSet.getLabels();

			printArray("Features", features);
			printArray("Labels", labels);
		}
		iterator.reset();
	}

	private void printArray(final String name, final INDArray array) {
		if (array.isMatrix()) {
			final double[][] matrix = array.toDoubleMatrix();
			logger.debug(name + " => columns: " + array.columns() + ", length: " + array.length() + ", matrix: ["
					+ matrix.length + "x" + matrix[0].length + "]");
		} else {
			final double[] vector = array.toDoubleVector();
			logger.debug(name + " => columns: " + array.columns() + ", length: " + array.length() + ", vector: ["
					+ vector.length + "]");
		}
	}

}
