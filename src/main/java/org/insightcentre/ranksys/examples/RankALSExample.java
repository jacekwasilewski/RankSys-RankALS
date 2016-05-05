/*
 * Copyright (C) 2016 Insight Centre for Data Analytics http://insight-centre.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.insightcentre.ranksys.examples;

import es.uam.eps.ir.ranksys.core.format.RecommendationFormat;
import es.uam.eps.ir.ranksys.core.format.SimpleRecommendationFormat;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;
import org.insightcentre.ranksys.mf.als.RankALSFactorizer;

import java.io.IOException;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;

import static es.uam.eps.ir.ranksys.core.util.parsing.DoubleParser.ddp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.lp;

public class RankALSExample {
    public static void main(String[] args) throws IOException {
        String userPath = args[0];
        String itemPath = args[1];
        String trainDataPath = args[2];
        String testDataPath = args[3];

        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(userPath, lp);
        FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(itemPath, lp);
        FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(trainDataPath, lp, lp, ddp, userIndex, itemIndex);
        FastPreferenceData<Long, Long> testData = SimpleFastPreferenceData.load(testDataPath, lp, lp, ddp, userIndex, itemIndex);

        // ranking matrix factorisation by Takács and Tikk
        int k = 20;
        int numIter = 10;

        Factorization<Long, Long> factorization = new RankALSFactorizer<Long, Long>(numIter).factorize(k, trainData);

        Recommender<Long, Long> recommender = new MFRecommender<>(userIndex, itemIndex, factorization);

        Set<Long> targetUsers = testData.getUsersWithPreferences().collect(Collectors.toSet());
        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
        Function<Long, IntPredicate> filter = FastFilters.notInTrain(trainData);
        int maxLength = 100;
        RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(userIndex, itemIndex, targetUsers, format, filter, maxLength);

        System.out.println("Running RankALS");
        runner.run(recommender, "rankals");
    }
}
