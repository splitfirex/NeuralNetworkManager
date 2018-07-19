package io.github.splitfirex;

import io.github.splitfirex.nn.DataSet;
import io.github.splitfirex.nn.NNetwork;
import io.github.splitfirex.utils.DataHelper;
import io.github.splitfirex.utils.PerformanceHelper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Main {
    public static void main(String... args) {
        List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("wine-q.csv"), ";", 1);
        // 6;0.21;0.38;0.8;0.02;22;98;0.98941;3.26;0.32;11.8;6
        //List<DataSet> dataset = DataHelper.toDataSet(DataHelper.normalize(data, IntStream.range(0,11).boxed().collect(Collectors.toList())), 10);
        List<DataSet> dataset = DataHelper.toDataSet(DataHelper.normalize(data, Arrays.asList(0, 11)), 11);


        IntStream.range(4, 20).parallel().forEach(x -> {
            NNetwork nn = new NNetwork(data.get(0).size() - 1, new int[]{x}, 6, 1, 0.9);
            nn.train(dataset, 1);
            nn.setIdNetwork(x - 4);

            List accum = new ArrayList<>();
            dataset.stream().forEach(y -> PerformanceHelper.efficency(accum, nn.compute(y.getInput()), y.getTarget()));
            System.out.print("ID: " + nn.getIdNetwork() + " ");
            PerformanceHelper.total(accum);

        });

       /* List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("wine.csv"), ",");
        List<DataSet> dataset = DataHelper.toDataSet(DataHelper.normalize(data, Arrays.asList()),0);
        NNetwork nn = new NNetwork(data.get(0).size() - 1, new int[]{2,6}, 3, 0.5, 0.9);
        nn.train(dataset, 0.1);
        List<Integer> accum = new ArrayList<>();
        dataset.stream().forEach(x->Performance.efficency(accum,nn.compute(x.getInput()),x.getTarget()));
        System.out.println(accum.stream().mapToDouble(x->x).summaryStatistics());*/
    }
}
