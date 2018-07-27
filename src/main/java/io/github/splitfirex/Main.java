package io.github.splitfirex;

import io.github.splitfirex.nn.DataSet;
import io.github.splitfirex.nn.NNetwork;
import io.github.splitfirex.utils.DataHelper;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String... args) {
        Integer skipheader = 1;

        List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("adult.csv"), ",");


        List<DataSet> dataset = DataHelper.toDataSet(DataHelper.normalize(data,
                data.get(0)
                        .stream()
                        .map(x -> Integer.parseInt(x))
                        .collect(Collectors.toCollection(ArrayList::new)), skipheader),
                Integer.parseInt(data.get(0).get(data.get(0).size() - 1)));

        NNetwork nn = new NNetwork(data.get(0 + skipheader).size() - 1, new int[]{10}, dataset.get(0).getTarget().length, 0.5, 0.9);
        nn.train(dataset, 99, 100);

       /* List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("wine.csv"), ",");
        List<DataSet> dataset = DataHelper.toDataSet(DataHelper.normalize(data, Arrays.asList()),0);
        NNetwork nn = new NNetwork(data.get(0).size() - 1, new int[]{2,6}, 3, 0.5, 0.9);
        nn.train(dataset, 0.1);
        List<Integer> accum = new ArrayList<>();
        dataset.stream().forEach(x->Performance.efficency(accum,nn.compute(x.getInput()),x.getTarget()));
        System.out.println(accum.stream().mapToDouble(x->x).summaryStatistics());*/
    }
}
