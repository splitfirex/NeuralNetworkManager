package io.github.splitfirex.utils;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DataHelper {

    public static List<List<String>> readFile(InputStream is, String separator) {

        List<List<String>> result = new ArrayList<>();

        try (Stream<String> lines = new BufferedReader(new InputStreamReader(is)).lines()) {
            lines.forEach(x -> result.add(Stream.of(x.split(separator)).collect(Collectors.toList())));
        }
        return result;
    }

}
