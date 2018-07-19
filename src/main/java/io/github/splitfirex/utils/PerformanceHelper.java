package io.github.splitfirex.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;


public class PerformanceHelper {

    public static void efficency(List<Integer> acuum, double[] output, double[] expected) {
        acuum.addAll(IntStream.range(0, output.length).map(x -> output[x] == expected[x] ? 1 : 0).collect(ArrayList::new, ArrayList::add, ArrayList::addAll));
    }

    public static void total(List<Integer> acuum) {
        System.out.println(acuum.stream().mapToDouble(x -> x).summaryStatistics());
    }

}
