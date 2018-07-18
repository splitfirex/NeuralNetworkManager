package io.github.splitfirex;

import io.github.splitfirex.nn.NNetwork;
import io.github.splitfirex.utils.DataHelper;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String... args) {
        List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("data.csv"), ",");
        System.out.println(DataHelper.normalize(data, Arrays.asList(4), 4));

        NNetwork nn = new NNetwork(data.get(0).size() - 1, new int[]{2, 2}, 3, 0.4, 0.9);
    }
}
