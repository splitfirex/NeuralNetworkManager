package io.github.splitfirex.utils;

import io.github.splitfirex.nn.DataSet;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DataHelper {

    public static List<List<String>> readFile(InputStream is, String separator, int skipheader) {

        List<List<String>> result = new ArrayList<>();

        try (Stream<String> lines = new BufferedReader(new InputStreamReader(is)).lines()) {
            lines.skip(skipheader).forEach(x -> result.add(Stream.of(x.split(separator)).collect(Collectors.toList())));
        }
        return result;
    }

    public static double[][] normalize(List<List<String>> input, List<Integer> dist) {

        double[][] output = new double[input.size()][input.get(0).size()];

        double[][] minmax = new double[input.get(0).size()][2];

        for (int i = 0; i < input.get(0).size(); i++) {
            minmax[i][0] = Integer.MAX_VALUE;
            minmax[i][1] = Integer.MIN_VALUE;
        }

        Map<Integer, Map<String, Integer>> discretes = new HashMap<Integer, Map<String, Integer>>();

        for (int i = 0; i < input.size(); i++) {
            for (int ii = 0; ii < input.get(i).size(); ii++) {
                if (dist.contains(new Integer(ii))) {
                    if (discretes.get(ii) == null)
                        discretes.put(ii, new HashMap<>());
                    if (!discretes.get(ii).keySet().contains(input.get(i).get(ii)))
                        discretes.get(ii).put(input.get(i).get(ii), discretes.get(ii).size());
                    continue;
                }
                //min
                minmax[ii][0] = minmax[ii][0] > Double.parseDouble(input.get(i).get(ii)) ? Double.parseDouble(input.get(i).get(ii)) : minmax[ii][0];

                //max
                minmax[ii][1] = minmax[ii][1] < Double.parseDouble(input.get(i).get(ii)) ? Double.parseDouble(input.get(i).get(ii)) : minmax[ii][1];
            }
        }

        for (int i = 0; i < input.size(); i++) {
            for (int ii = 0; ii < input.get(i).size(); ii++) {
                if (dist.contains(new Integer(ii))) {
                    output[i][ii] = discretes.get(ii).get(input.get(i).get(ii));
                    continue;
                }
                output[i][ii] = (Double.parseDouble(input.get(i).get(ii)) - minmax[ii][0]) / (minmax[ii][1] - minmax[ii][0]);
            }
        }
        return output;
    }

    public static List<DataSet> toDataSet(double[][] data, int outputColumn) {

        AtomicInteger i = new AtomicInteger(0);
        List<DataSet> output = new ArrayList<>();
        Map<Double, Integer> discretes = new HashMap<>();
        Stream.of(data).forEach(x -> {
            if (discretes.get(x[outputColumn]) == null) {
                discretes.put(x[outputColumn], i.getAndIncrement());
            }
        });

        Stream.of(data).forEach(x ->
                output.add(
                        new DataSet(
                                IntStream.range(0, x.length).filter(y -> y != outputColumn).mapToDouble(z -> x[z]).toArray(),
                                IntStream.range(0, discretes.size()).mapToDouble(y ->
                                        y == discretes.get(x[outputColumn]) ? 1.0 : 0.0
                                ).toArray())
                )
        );
        return output;
    }

}
