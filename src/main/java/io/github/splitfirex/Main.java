package io.github.splitfirex;

import io.github.splitfirex.utils.DataHelper;

import java.util.List;

public class Main {
    public static void main(String... args) {
        List<List<String>> data = DataHelper.readFile(ClassLoader.getSystemResourceAsStream("data.csv"), ",");
        System.out.println(data);
    }
}
