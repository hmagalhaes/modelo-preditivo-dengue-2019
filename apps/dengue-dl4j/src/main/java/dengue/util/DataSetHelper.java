package dengue.util;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DataSetHelper {

	public static List<Double> getLabels(final DataSetIterator iterator) {
		final List<Double> labelList = new ArrayList<>();

		iterator.reset();
		while (iterator.hasNext()) {
			final DataSet dataSet = iterator.next();
			final double[] labels = dataSet.getLabels().toDoubleVector();

			add(labels, labelList);
		}
		iterator.reset();

		return labelList;
	}

	private static void add(final double[] source, final List<Double> target) {
		for (double value : source) {
			target.add(value);
		}
	}

	public static List<Writable> toWritable(final List<Double> doubleList) {
		return doubleList.stream().map(value -> new DoubleWritable(value)).collect(Collectors.toList());
	}

}
