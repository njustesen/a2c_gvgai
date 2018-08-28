class AStar {
    constructor(empty) {
        this._empty = empty;
    }
    insideMap(map, p) {
        return p.x >= 0 && p.y >= 0 && p.x <= map[0].length - 1 && p.y <= map.length - 1;
    }
    passable(map, p) {
        return this._empty.indexOf(map[p.y][p.x]) != -1;
    }
    pathExisits(map, start, end) {
        if (!this.insideMap(map, start) || !this.insideMap(map, end) ||
            !this.passable(map, start) || !this.passable(map, end)) {
            return false;
        }
        let visited = {};
        let queue = [{ x: start.x, y: start.y }];
        while (queue.length > 0) {
            queue.sort((a, b) => {
                let dist1 = Math.abs(a.x - end.x) + Math.abs(a.y - end.y);
                let dist2 = Math.abs(b.x - end.x) + Math.abs(b.y - end.y);
                return dist1 - dist2;
            });
            let currentNode = queue.splice(0, 1)[0];
            if (currentNode.x == end.x && currentNode.y == end.y) {
                return true;
            }
            if (!visited.hasOwnProperty(currentNode.x + "," + currentNode.y)) {
                visited[currentNode.x + "," + currentNode.y] = true;
                for (let dx = -1; dx <= 1; dx++) {
                    for (let dy = -1; dy <= 1; dy++) {
                        if ((dx == 0 || dy == 0) && !(dx == 0 && dy == 0) &&
                            this.insideMap(map, { x: currentNode.x + dx, y: currentNode.y + dy }) &&
                            this.passable(map, { x: currentNode.x + dx, y: currentNode.y + dy })) {
                            queue.push({ x: currentNode.x + dx, y: currentNode.y + dy });
                        }
                    }
                }
            }
        }
        return false;
    }
}
/// <reference path="../AStar.ts"/>
class WaitForBreakfast {
    constructor(aStar, charMap) {
        this._aStar = aStar;
        this._charMap = charMap;
    }
    getDistance(p1, p2) {
        return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
    }
    getRandomLocation(width, height) {
        return {
            x: Math.floor(Math.random() * width) + 1,
            y: Math.floor(Math.random() * height) + 1
        };
    }
    placeTable(map, start, end) {
        let dx = Math.sign(end.x - start.x);
        let dy = Math.sign(end.y - start.y);
        let current = { x: start.x, y: start.y };
        while (this.getDistance(current, end) != 0) {
            map[current.y][current.x] = 2;
            current.x += dx;
            current.y += dy;
        }
    }
    removeTable(map, start, end) {
        let dx = Math.sign(end.x - start.x);
        let dy = Math.sign(end.y - start.y);
        let current = { x: start.x, y: start.y };
        while (this.getDistance(current, end) != 0) {
            map[current.y][current.x] = 0;
            current.x += dx;
            current.y += dy;
        }
    }
    putTables(map, start, waiter, target, maxNumber) {
        maxNumber = Math.min(maxNumber, Math.floor(map.length / 2), Math.floor(map[0].length / 2));
        let length = Math.floor(Math.random() * maxNumber) + 1;
        let location = this.getRandomLocation(map[0].length - length - 2, map.length - length - 2);
        if (Math.random() < 0.4) {
            length = 1;
        }
        let dir = Math.round(Math.random());
        this.placeTable(map, location, { x: location.x + dir * length, y: location.y + (1 - dir) * length });
        while (!this._aStar.pathExisits(map, start, target) ||
            !this._aStar.pathExisits(map, target, waiter)) {
            this.removeTable(map, location, { x: location.x + dir * length, y: location.y + (1 - dir) * length });
            length = Math.floor(Math.random() * maxNumber) + 1;
            if (Math.random() < 0.4) {
                length = 1;
            }
            dir = Math.round(Math.random());
            location = this.getRandomLocation(map[0].length - length - 2, map.length - length - 2);
            this.placeTable(map, location, { x: location.x + dir * length, y: location.y + (1 - dir) * length });
        }
        return length;
    }
    shuffleArray(array) {
        for (let i = 0; i < array.length; i++) {
            let index = Math.floor(Math.random() * array.length);
            let temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
    placeChairs(map, table, index) {
        let points = [];
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                if ((x == 0 || y == 0) && !(x == 0 && y == 0)) {
                    points.push({ x: table.x + x, y: table.y + y });
                }
            }
        }
        this.shuffleArray(points);
        let prob = 1;
        for (let p of points) {
            if (map[p.y][p.x] == 0) {
                if (Math.random() < prob) {
                    map[p.y][p.x] = index;
                }
                prob /= 2;
            }
        }
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        let width = Math.floor((maxWidth - 8) * diff + 3 * Math.random()) + 5;
        let height = Math.floor((maxHeight - 8) * diff + 3 * Math.random()) + 5;
        let tableNumbers = 0.7 * diff + 0.3 * Math.random();
        return [width, height, tableNumbers];
    }
    adjustParameters(width, height, tableNumbers) {
        let parameters = [tableNumbers];
        parameters[0] = Math.floor(tableNumbers * 0.1 * width * height);
        return [Math.max(width, 5), Math.max(height, 5)].concat(parameters);
    }
    generate(roomWidth, roomHeight, tableNumbers) {
        let map = [];
        for (let y = 0; y < roomHeight; y++) {
            map.push([]);
            for (let x = 0; x < roomWidth; x++) {
                if (x == 0 || y == 0 || y == roomHeight - 1 || x == roomWidth - 1) {
                    map[y].push(1);
                }
                else {
                    map[y].push(0);
                }
            }
        }
        let start = {
            x: Math.round(Math.random()) * (roomWidth - 1),
            y: Math.floor(Math.random() * (roomHeight - 2)) + 1
        };
        if (Math.random() < 0.5) {
            start = {
                x: Math.floor(Math.random() * (roomWidth - 2)) + 1,
                y: Math.round(Math.random()) * (roomHeight - 1)
            };
        }
        map[start.y][start.x] = 0;
        let waiter = {
            x: Math.round(Math.random()) * (roomWidth - 1),
            y: Math.floor(Math.random() * (roomHeight - 2)) + 1
        };
        if (Math.random() < 0.5) {
            waiter = {
                x: Math.floor(Math.random() * (roomWidth - 2)) + 1,
                y: Math.round(Math.random()) * (roomHeight - 1)
            };
        }
        while (this.getDistance(start, waiter) < Math.min(roomWidth, roomHeight)) {
            waiter = {
                x: Math.round(Math.random()) * (roomWidth - 1),
                y: Math.floor(Math.random() * (roomHeight - 2)) + 1
            };
            if (Math.random() < 0.5) {
                waiter = {
                    x: Math.floor(Math.random() * (roomWidth - 2)) + 1,
                    y: Math.round(Math.random()) * (roomHeight - 1)
                };
            }
        }
        map[waiter.y][waiter.x] = 0;
        let target = this.getRandomLocation(roomWidth - 2, roomHeight - 2);
        while (this.getDistance(start, target) + this.getDistance(target, waiter) < Math.min(roomWidth, roomHeight) &&
            this.getDistance(start, target) < Math.min(roomWidth, roomHeight) / 2 &&
            this.getDistance(target, waiter) < Math.min(roomWidth, roomHeight) / 2) {
            target = this.getRandomLocation(roomWidth - 2, roomHeight - 2);
        }
        while (tableNumbers > 0) {
            let length = this.putTables(map, start, waiter, target, tableNumbers);
            if (length == 0) {
                break;
            }
            tableNumbers -= length;
        }
        map[target.y][target.x] = 3;
        map[waiter.y][waiter.x] = 4;
        map[start.y][start.x] = 5;
        if (start.x == 0) {
            map[start.y][1] = 6;
        }
        if (start.x == roomWidth - 1) {
            map[start.y][roomWidth - 2] = 6;
        }
        if (start.y == 0) {
            map[1][start.x] = 6;
        }
        if (start.y == roomHeight - 1) {
            map[roomHeight - 2][start.x] = 6;
        }
        for (let y = 0; y < map.length; y++) {
            for (let x = 0; x < map[y].length; x++) {
                if (map[y][x] == 2) {
                    this.placeChairs(map, { x: x, y: y }, 7);
                }
                if (map[y][x] == 3) {
                    this.placeChairs(map, { x: x, y: y }, 8);
                }
            }
        }
        let targetChair = false;
        let levelString = "";
        for (let y = 0; y < roomHeight; y++) {
            for (let x = 0; x < roomWidth; x++) {
                switch (map[y][x]) {
                    case 0:
                        levelString += this._charMap[WaitForBreakfast.EMPTY];
                        break;
                    case 1:
                        levelString += this._charMap[WaitForBreakfast.WALL];
                        break;
                    case 2:
                        levelString += this._charMap[WaitForBreakfast.TABLE];
                        break;
                    case 3:
                        levelString += this._charMap[WaitForBreakfast.TARGET_TABLE];
                        break;
                    case 4:
                        levelString += this._charMap[WaitForBreakfast.WAITER];
                        break;
                    case 5:
                        levelString += this._charMap[WaitForBreakfast.EXIT];
                        break;
                    case 6:
                        levelString += this._charMap[WaitForBreakfast.AVATAR];
                        break;
                    case 7:
                        if (map[y + 1][x] == 2) {
                            levelString += this._charMap[WaitForBreakfast.UP];
                        }
                        else if (map[y - 1][x] == 2) {
                            levelString += this._charMap[WaitForBreakfast.DOWN];
                        }
                        else if (map[y][x + 1] == 2) {
                            levelString += this._charMap[WaitForBreakfast.LEFT];
                        }
                        else if (map[y][x - 1] == 2) {
                            levelString += this._charMap[WaitForBreakfast.RIGHT];
                        }
                        break;
                    case 8:
                        if (targetChair) {
                            levelString += this._charMap[WaitForBreakfast.EMPTY];
                            continue;
                        }
                        targetChair = true;
                        if (map[y + 1][x] == 3) {
                            levelString += this._charMap[WaitForBreakfast.TARGET_UP];
                        }
                        else if (map[y - 1][x] == 3) {
                            levelString += this._charMap[WaitForBreakfast.TARGET_DOWN];
                        }
                        else if (map[y][x + 1] == 3) {
                            levelString += this._charMap[WaitForBreakfast.TARGET_LEFT];
                        }
                        else if (map[y][x - 1] == 3) {
                            levelString += this._charMap[WaitForBreakfast.TARGET_RIGHT];
                        }
                        break;
                }
            }
            levelString += "\n";
        }
        return levelString;
    }
}
WaitForBreakfast.EMPTY = "empty";
WaitForBreakfast.WALL = "wall";
WaitForBreakfast.EXIT = "exit";
WaitForBreakfast.WAITER = "waiter";
WaitForBreakfast.TARGET_TABLE = "targetTable";
WaitForBreakfast.TARGET_LEFT = "targetLeft";
WaitForBreakfast.TARGET_RIGHT = "targetRight";
WaitForBreakfast.TARGET_UP = "targetUp";
WaitForBreakfast.TARGET_DOWN = "targetDown";
WaitForBreakfast.TABLE = "table";
WaitForBreakfast.LEFT = "left";
WaitForBreakfast.RIGHT = "right";
WaitForBreakfast.UP = "up";
WaitForBreakfast.DOWN = "down";
WaitForBreakfast.AVATAR = "avatar";
class ZenPuzzle {
    constructor(hilbert2D, charMap) {
        this._hilbert2D = hilbert2D;
        this._charMap = charMap;
    }
    sign(value) {
        if (value > 0) {
            return 1;
        }
        if (value < 0) {
            return -1;
        }
        return 0;
    }
    putZeroes(map, start, end) {
        if ((start.x > map[0].length - 1 || start.y > map.length - 1) &&
            (end.x > map[0].length - 1 || end.y > map.length - 1)) {
            return;
        }
        if (start.x > map[0].length - 1 || start.y > map.length - 1) {
            let temp = start;
            start = end;
            end = temp;
        }
        if (Math.abs(start.x - end.x) == 2 && start.y - end.y == 0) {
            let dx = this.sign(end.x - start.x);
            map[start.y][start.x] = 0;
            map[start.y][Math.min(start.x + dx, map[0].length - 1)] = 0;
            map[start.y][Math.min(start.x + 2 * dx, map[0].length - 1)] = 0;
        }
        if (Math.abs(start.y - end.y) == 2 && start.x - end.x == 0) {
            let dy = this.sign(end.y - start.y);
            map[start.y][start.x] = 0;
            map[Math.min(map.length - 1, start.y + dy)][start.x] = 0;
            map[Math.min(map.length - 1, start.y + 2 * dy)][start.x] = 0;
        }
    }
    appendEmpty(rows, columns) {
        let lines = "";
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < columns; j++) {
                lines += this._charMap[ZenPuzzle.EMPTY];
            }
            lines += "\n";
        }
        return lines;
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        let width = Math.floor(Math.max(0, maxWidth - 10) * diff + 2 * Math.random()) + 2;
        let height = Math.floor(Math.max(0, maxHeight - 8) * diff + 2 * Math.random()) + 2;
        let borderX = (0.3 * diff + 0.4 * Math.random() + 0.3);
        let borderY = (0.3 * diff + 0.4 * Math.random() + 0.3);
        return [width, height, borderX, borderY];
    }
    adjustParameters(width, height, borderX = 0, borderY = 0) {
        let parameters = [borderX, borderY];
        parameters[0] = 2 * Math.floor(borderX * width / 4);
        parameters[1] = 2 * Math.floor(borderY * height / 4);
        return [Math.max(width, 2), Math.max(height, 2)].concat(parameters);
    }
    generate(boardWidth, boardHeight, borderX = 0, borderY = 0) {
        let curveSize = Math.pow(2, Math.ceil(Math.max(Math.log2(boardWidth), Math.log2(boardHeight))));
        let h2d = new this._hilbert2D(curveSize);
        let start = {
            x: Math.floor(Math.random() * (curveSize - boardWidth / 2)),
            y: Math.floor(Math.random() * (curveSize - boardHeight / 2))
        };
        let end = {
            x: start.x + Math.floor(boardWidth / 2) + 2,
            y: start.y + Math.floor(boardHeight / 2) + 2
        };
        let points = [];
        for (let i = 0; i < curveSize * curveSize; i++) {
            let p = h2d.xy(i);
            if (p.x >= start.x && p.x < end.x && p.y >= start.y && p.y < end.y) {
                points.push({ x: 2 * (p.x - start.x), y: 2 * (p.y - start.y) });
            }
        }
        let results = [];
        for (let y = 0; y < boardHeight; y++) {
            results.push([]);
            for (let x = 0; x < boardWidth; x++) {
                if (x >= borderX && y >= borderY &&
                    x <= boardWidth - borderX - 1 && y <= boardHeight - borderY - 1) {
                    results[y].push(1);
                }
                else {
                    results[y].push(0);
                }
            }
        }
        for (let i = 0; i < points.length; i++) {
            if (i < points.length - 1) {
                this.putZeroes(results, points[i], points[i + 1]);
            }
        }
        let levelString = this.appendEmpty(2, results[0].length + 6);
        for (let y = 0; y < results.length; y++) {
            levelString += this._charMap[ZenPuzzle.EMPTY];
            if (y == Math.floor(results.length / 2)) {
                levelString += this._charMap[ZenPuzzle.AVATAR];
            }
            else {
                levelString += this._charMap[ZenPuzzle.EMPTY];
            }
            levelString += this._charMap[ZenPuzzle.EMPTY];
            for (let x = 0; x < results[y].length; x++) {
                if (results[y][x] == 0) {
                    levelString += this._charMap[ZenPuzzle.TILE];
                }
                else {
                    levelString += this._charMap[ZenPuzzle.ROCK];
                }
            }
            levelString += this._charMap[ZenPuzzle.EMPTY] + this._charMap[ZenPuzzle.EMPTY] +
                this._charMap[ZenPuzzle.EMPTY] + "\n";
        }
        levelString += this.appendEmpty(2, results[0].length + 6);
        levelString = levelString.substr(0, levelString.length - 1);
        return levelString;
    }
}
ZenPuzzle.AVATAR = "avatar";
ZenPuzzle.ROCK = "rock";
ZenPuzzle.TILE = "tile";
ZenPuzzle.EMPTY = "empty";
class CellularAutomata {
    constructor(solidFlipNum, emptyFlipNum, tinySolidNum) {
        this._solidFlipNum = solidFlipNum;
        this._emptyFlipNum = emptyFlipNum;
        this._tinySolidNum = tinySolidNum;
    }
    checkNumSolid(map, p, four = false) {
        let value = 0;
        for (let y = -1; y <= 1; y++) {
            for (let x = -1; x <= 1; x++) {
                if (x == 0 && y == 0) {
                    continue;
                }
                if (map[p.y + y][p.x + x] == 1) {
                    if (four) {
                        if (x == 0 || y == 0) {
                            value += 1;
                        }
                    }
                    else {
                        value += 1;
                    }
                }
            }
        }
        return value;
    }
    getUnlabeledLocation(map) {
        for (let y = 1; y < map.length - 1; y++) {
            for (let x = 1; x < map[y].length - 1; x++) {
                if (map[y][x] == 0) {
                    return { x: x, y: y };
                }
            }
        }
        return null;
    }
    labelMap(map, start, label) {
        let result = [];
        let locations = [start];
        while (locations.length > 0) {
            let current = locations.splice(0, 1)[0];
            if (map[current.y][current.x] == 0) {
                result.push(current);
                map[current.y][current.x] = label;
                for (let y = -1; y <= 1; y++) {
                    for (let x = -1; x <= 1; x++) {
                        if ((y == 0 || x == 0) && !(y == 0 && x == 0)) {
                            locations.push({ x: current.x + x, y: current.y + y });
                        }
                    }
                }
            }
        }
        return result;
    }
    getEmptyRegions(map) {
        let regions = [];
        let start = this.getUnlabeledLocation(map);
        let label = 2;
        while (start != null) {
            regions.push(this.labelMap(map, start, label));
            label += 1;
            start = this.getUnlabeledLocation(map);
        }
        return regions;
    }
    checkNumEmpty(map, p, four = false) {
        return 8 - this.checkNumSolid(map, p, four);
    }
    tinySolid(map) {
        for (let y = 1; y < map.length - 1; y++) {
            for (let x = 1; x < map[y].length - 1; x++) {
                if (map[y][x] == 1 && this.checkNumSolid(map, { x: x, y: y }, true) <= this._tinySolidNum) {
                    return true;
                }
            }
        }
        return false;
    }
    connect(map, p1, p2) {
        let dx = Math.sign(p2.x - p1.x);
        let dy = Math.sign(p2.y - p1.y);
        let x = p1.x;
        while (x != p2.x) {
            map[p1.y][x] = 0;
            x += dx;
        }
        let y = p1.y;
        while (y != p2.y) {
            map[y][p2.x] = 0;
            y += dy;
        }
    }
    clone(map) {
        let result = [];
        for (let y = 0; y < map.length; y++) {
            result.push([]);
            for (let x = 0; x < map[y].length; x++) {
                if (map[y][x] == 1) {
                    result[y].push(1);
                }
                else {
                    result[y].push(0);
                }
            }
        }
        return result;
    }
    generate(width, height, solidPercentage, iterations) {
        let result = [];
        for (let y = 0; y < height; y++) {
            result.push([]);
            for (let x = 0; x < width; x++) {
                if (Math.random() < solidPercentage ||
                    (y == 0 || x == 0 || y == height - 1 || x == width - 1)) {
                    result[y].push(1);
                }
                else {
                    result[y].push(0);
                }
            }
        }
        let i = 0;
        while ((this.tinySolid(result) && this._tinySolidNum >= 0) || i < iterations) {
            let newMap = this.clone(result);
            for (let y = 1; y < result.length - 1; y++) {
                for (let x = 1; x < result[y].length - 1; x++) {
                    if (result[y][x] == 0) {
                        if (this.checkNumSolid(result, { x: x, y: y }) > this._solidFlipNum) {
                            newMap[y][x] = 1;
                        }
                    }
                    else {
                        if (this.checkNumEmpty(result, { x: x, y: y }) > this._emptyFlipNum) {
                            newMap[y][x] = 0;
                        }
                    }
                }
            }
            result = newMap;
            i++;
        }
        let regions = this.getEmptyRegions(this.clone(result));
        for (let r1 of regions) {
            for (let r2 of regions) {
                let p1 = r1[Math.floor(Math.random() * r1.length)];
                let p2 = r2[Math.floor(Math.random() * r2.length)];
                this.connect(result, p1, p2);
            }
        }
        return result;
    }
}
/// <reference path="../CellularAutomata.ts"/>
/// <reference path="../AStar.ts"/>
class Boulderdash {
    constructor(ca, aStar, charMap) {
        this._ca = ca;
        this._aStar = aStar;
        this._charMap = charMap;
    }
    getRandomEmptyPlace(map) {
        let p = { x: 0, y: 0 };
        while (map[p.y][p.x] != 0) {
            p.x = Math.floor(Math.random() * map[0].length);
            p.y = Math.floor(Math.random() * map.length);
        }
        return p;
    }
    getAllPossibleLocations(map) {
        let possibleLocations = [];
        for (let x = 0; x < map[0].length; x++) {
            for (let y = 0; y < map.length; y++) {
                if (map[y][x] == 0) {
                    possibleLocations.push({ x: x, y: y });
                }
            }
        }
        return possibleLocations;
    }
    distance(p1, p2) {
        return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
    }
    reachable(map, start, locations) {
        for (let loc of locations) {
            if (!this._aStar.pathExisits(map, start, loc)) {
                return false;
            }
        }
        return true;
    }
    distanceToGems(loc, gems) {
        let dist = -1;
        for (let g of gems) {
            if (dist == -1 || this.distance(loc, g) < dist) {
                dist = this.distance(loc, g);
            }
        }
        return dist;
    }
    assignEmptyArea(map, start, length, dir) {
        let dx = dir * (2 * Math.round(Math.random()) - 1);
        let dy = (1 - dir) * (2 * Math.round(Math.random()) - 1);
        let current = { x: start.x + dx, y: start.y + dy };
        for (let i = 0; i < length; i++) {
            if (map[current.y][current.x] == 0) {
                map[current.y][current.x] = 6;
            }
            else {
                break;
            }
            current.x += dx;
            current.y += dy;
        }
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        let width = Math.floor((maxWidth - 12) * diff + 2 * Math.random()) + 10;
        let height = Math.floor((maxHeight - 12) * diff + 2 * Math.random()) + 10;
        let solidPercentage = 0.6 * diff + 0.2 + 0.2 * Math.random();
        let smoothness = 0.5 * (1 - diff) + 0.3 + 0.2 * Math.random();
        let enemies = (0.7 + 0.3 * Math.random()) * diff;
        let boulders = (0.7 + 0.3 * Math.random()) * diff;
        let extraGems = 0.7 * (1 - diff) + 0.1 + 0.2 * Math.random();
        return [width, height, solidPercentage, smoothness, enemies, boulders, extraGems];
    }
    adjustParameters(width, height, solidPercentage, smoothness, enemies, boulders, extraGems) {
        let parameters = [solidPercentage, smoothness, enemies, boulders, extraGems];
        parameters[0] = solidPercentage * 0.5;
        parameters[1] = Math.floor(smoothness * 10 + 1);
        parameters[2] = Math.floor(enemies * 0.05 * width * height);
        parameters[4] = Math.floor(extraGems * 0.05 * width * height);
        parameters[3] = Math.floor(boulders * 2 * (10 + extraGems));
        return [Math.max(width, 10), Math.max(height, 10)].concat(parameters);
    }
    generate(width, height, solidPercentage, smoothness, enemies, boulders, extraGems) {
        let map = this._ca.generate(width, height, solidPercentage, smoothness);
        let start = this.getRandomEmptyPlace(map);
        let exit = this.getRandomEmptyPlace(map);
        while (this.distance(start, exit) < 4) {
            exit = this.getRandomEmptyPlace(map);
        }
        let possibleLocations = this.getAllPossibleLocations(map);
        possibleLocations.sort((a, b) => { return 2 * Math.random() - 1; });
        let gems = [];
        for (let p of possibleLocations) {
            if (gems.length == 10 + extraGems) {
                break;
            }
            if (!((p.x == start.x && p.y == start.y) || (p.x == exit.x && p.y == exit.y))) {
                gems.push(p);
                map[p.y][p.x] = 2;
            }
        }
        possibleLocations = this.getAllPossibleLocations(map);
        possibleLocations.sort((a, b) => { return this.distanceToGems(a, gems) - this.distanceToGems(b, gems) + 0.1 * (2 * Math.random() - 1); });
        for (let p of possibleLocations) {
            if (boulders == 0) {
                break;
            }
            if (!(this.distance(p, start) > 4 && this.distance(p, exit) > 4)) {
                continue;
            }
            map[p.y][p.x] = 3;
            if (this.reachable(map, start, gems.concat([exit]))) {
                boulders -= 1;
            }
            else {
                map[p.y][p.x] = 0;
            }
        }
        possibleLocations = this.getAllPossibleLocations(map);
        possibleLocations.sort((a, b) => { return 2 * Math.random() - 1; });
        for (let p of possibleLocations) {
            if (enemies == 0) {
                break;
            }
            if (this.distance(p, start) > 5 && this.distance(p, exit) > 2) {
                map[p.y][p.x] = 4 + Math.round(Math.random());
                enemies -= 1;
                this.assignEmptyArea(map, p, Math.floor(5 * Math.random()), Math.round(Math.random()));
            }
        }
        map[start.y][start.x] = 7;
        map[exit.y][exit.x] = 8;
        let result = "";
        for (let y = 0; y < map.length; y++) {
            for (let x = 0; x < map[y].length; x++) {
                switch (map[y][x]) {
                    case 0:
                        result += this._charMap[Boulderdash.DIRT];
                        break;
                    case 1:
                        result += this._charMap[Boulderdash.WALL];
                        break;
                    case 2:
                        result += this._charMap[Boulderdash.GEM];
                        break;
                    case 3:
                        result += this._charMap[Boulderdash.BOULDER];
                        break;
                    case 4:
                        result += this._charMap[Boulderdash.BUTTERFLY];
                        break;
                    case 5:
                        result += this._charMap[Boulderdash.CRAB];
                        break;
                    case 6:
                        result += this._charMap[Boulderdash.EMPTY];
                        break;
                    case 7:
                        result += this._charMap[Boulderdash.AVATAR];
                        break;
                    case 8:
                        result += this._charMap[Boulderdash.EXIT];
                        break;
                }
            }
            result += "\n";
        }
        return result;
    }
}
Boulderdash.WALL = "wall";
Boulderdash.DIRT = "dirt";
Boulderdash.EMPTY = "empty";
Boulderdash.CRAB = "crab";
Boulderdash.BUTTERFLY = "butterfly";
Boulderdash.BOULDER = "boulder";
Boulderdash.AVATAR = "avatar";
Boulderdash.GEM = "gem";
Boulderdash.EXIT = "exit";
class Cell {
    constructor() {
        this._walls = [true, true, true, true];
        this.marked = false;
    }
    unlockDirection(dir) {
        if (dir.x == -1) {
            this._walls[0] = false;
        }
        if (dir.x == 1) {
            this._walls[1] = false;
        }
        if (dir.y == -1) {
            this._walls[2] = false;
        }
        if (dir.y == 1) {
            this._walls[3] = false;
        }
    }
    getWall(dir) {
        if (dir.x == -1) {
            return this._walls[0];
        }
        if (dir.x == 1) {
            return this._walls[1];
        }
        if (dir.y == -1) {
            return this._walls[2];
        }
        if (dir.y == 1) {
            return this._walls[3];
        }
        return true;
    }
}
class Maze {
    constructor() {
    }
    generate(width, height) {
        let maze = [];
        for (let y = 0; y < height; y++) {
            maze.push([]);
            for (let x = 0; x < width; x++) {
                maze[y].push(new Cell());
            }
        }
        let start = { x: Math.floor(Math.random() * width), y: Math.floor(Math.random() * height) };
        let open = [start];
        while (open.length > 0) {
            open.sort((a, b) => { return 2 * Math.random() - 1; });
            let current = open.splice(0, 1)[0];
            if (!maze[current.y][current.x].marked) {
                let surrounding = [];
                for (let x = -1; x <= 1; x++) {
                    for (let y = -1; y <= 1; y++) {
                        if ((x == 0 || y == 0) && !(x == 0 && y == 0)) {
                            let newPos = { x: current.x + x, y: current.y + y };
                            if (newPos.x >= 0 && newPos.y >= 0 && newPos.x <= width - 1 && newPos.y <= height - 1) {
                                if (maze[newPos.y][newPos.x].marked) {
                                    surrounding.push({ x: x, y: y });
                                }
                            }
                        }
                    }
                }
                surrounding.sort((a, b) => { return Math.random() - 0.5; });
                if (surrounding.length > 0) {
                    maze[current.y][current.x].unlockDirection(surrounding[0]);
                    maze[current.y + surrounding[0].y][current.x + surrounding[0].x].unlockDirection({ x: -1 * surrounding[0].x, y: -1 * surrounding[0].y });
                }
                maze[current.y][current.x].marked = true;
                for (let x = -1; x <= 1; x++) {
                    for (let y = -1; y <= 1; y++) {
                        if ((x == 0 || y == 0) && !(x == 0 && y == 0)) {
                            let newPos = { x: current.x + x, y: current.y + y };
                            if (newPos.x >= 0 && newPos.y >= 0 && newPos.x <= width - 1 && newPos.y <= height - 1) {
                                open.push(newPos);
                            }
                        }
                    }
                }
            }
        }
        let result = [];
        for (let y = 0; y < 2 * height + 1; y++) {
            result.push([]);
            for (let x = 0; x < 2 * width + 1; x++) {
                result[y].push(1);
            }
        }
        for (let y = 0; y < result.length; y++) {
            for (let x = 0; x < result[y].length; x++) {
                if (y % 2 == 1 && x % 2 == 1) {
                    let pos = { x: Math.floor(x / 2), y: Math.floor(y / 2) };
                    result[y][x] = 0;
                    if (!maze[pos.y][pos.x].getWall({ x: -1, y: 0 })) {
                        result[y][x - 1] = 0;
                    }
                    if (!maze[pos.y][pos.x].getWall({ x: 1, y: 0 })) {
                        result[y][x + 1] = 0;
                    }
                    if (!maze[pos.y][pos.x].getWall({ x: 0, y: -1 })) {
                        result[y - 1][x] = 0;
                    }
                    if (!maze[pos.y][pos.x].getWall({ x: 0, y: 1 })) {
                        result[y + 1][x] = 0;
                    }
                }
            }
        }
        return result;
    }
}
/// <reference path="../Maze.ts"/>
class Zelda {
    constructor(maze, charMap) {
        this._maze = maze;
        this._charMap = charMap;
    }
    getAllPossibleLocations(map) {
        let possibleLocations = [];
        for (let x = 0; x < map[0].length; x++) {
            for (let y = 0; y < map.length; y++) {
                if (map[y][x] == 0) {
                    possibleLocations.push({ x: x, y: y });
                }
            }
        }
        return possibleLocations;
    }
    getAllSeparatorWalls(map) {
        let possibleLocations = [];
        for (let x = 1; x < map[0].length - 1; x++) {
            for (let y = 1; y < map.length - 1; y++) {
                if (map[y][x] == 1 && ((map[y - 1][x] == 0 && map[y + 1][x] == 0) || (map[y][x - 1] == 0 && map[y][x + 1] == 0))) {
                    possibleLocations.push({ x: x, y: y });
                }
            }
        }
        return possibleLocations;
    }
    distance(p1, p2) {
        return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        // let width: number = Math.floor(diff * Math.max(maxWidth - 6, 0) + 2 * Math.random()) + 4;
        let width = maxWidth;
        // let height: number = Math.floor(diff * Math.max(maxHeight - 6, 0) + 2 * Math.random()) + 4;
        let height = maxHeight;
        // let openess:number = (1 - diff) * 0.4 + 0.1 * Math.random();
        let openess = (1 - diff) * 0.3 + 0.1 * Math.random() + 0.1;
        // let enemies:number = diff * 0.5 + 0.1 * Math.random() + 0.4;
        let enemies = diff * 0.4 + 0.2 * Math.random();
        let distanceToGoal = diff * 0.7 + 0.3 * Math.random();
        return [width, height, openess, enemies, distanceToGoal];
    }
    adjustParameters(width, height, openess, enemies, distanceToGoal) {
        let parameters = [openess, enemies, distanceToGoal];
        parameters[0] = Math.floor(openess * (width - 1) * (height - 1));
        parameters[1] = Math.floor(enemies * 0.05 * width * height);
        parameters[2] = distanceToGoal + 1;
        return [Math.max(width, 4), Math.max(height, 4)].concat(parameters);
    }
    generate(width, height, openess, enemies, distanceToGoal) {
        let map = this._maze.generate(Math.floor(width / 2), Math.floor(height / 2));
        let walls = this.getAllSeparatorWalls(map);
        walls.sort((a, b) => { return Math.random() - 0.5; });
        for (let i = 0; i < walls.length; i++) {
            if (openess == 0) {
                break;
            }
            map[walls[i].y][walls[i].x] = 0;
            openess -= 1;
        }
        let locations = this.getAllPossibleLocations(map);
        locations.sort((a, b) => { return Math.random() - 0.5; });
        let avatar = locations.splice(0, 1)[0];
        map[avatar.y][avatar.x] = 2;
        locations.sort((a, b) => {
            return this.distance(avatar, b) - distanceToGoal * this.distance(avatar, a) +
                Math.min(width, height) * (2 * Math.random() - 1);
        });
        let exit = locations.splice(0, 1)[0];
        map[exit.y][exit.x] = 3;
        locations.sort((a, b) => {
            return this.distance(avatar, b) - distanceToGoal * this.distance(avatar, a) +
                this.distance(exit, b) - this.distance(exit, a) +
                Math.min(width, height) * (2 * Math.random() - 1);
        });
        let key = locations.splice(0, 1)[0];
        map[key.y][key.x] = 4;
        locations.sort((a, b) => {
            return this.distance(avatar, b) - this.distance(avatar, a) +
                Math.min(width, height) * (2 * Math.random() - 1);
        });
        for (let l of locations) {
            if (enemies == 0) {
                break;
            }
            map[l.y][l.x] = 5 + Math.floor(Math.random() * 3);
            enemies -= 1;
        }
        let result = "";
        for (let y = 0; y < map.length; y++) {
            for (let x = 0; x < map[y].length; x++) {
                switch (map[y][x]) {
                    case 0:
                        result += this._charMap[Zelda.EMPTY];
                        break;
                    case 1:
                        result += this._charMap[Zelda.WALL];
                        break;
                    case 2:
                        result += this._charMap[Zelda.AVATAR];
                        break;
                    case 3:
                        result += this._charMap[Zelda.EXIT];
                        break;
                    case 4:
                        result += this._charMap[Zelda.KEY];
                        break;
                    case 5:
                        result += this._charMap[Zelda.MONSTER_SLOW];
                        break;
                    case 6:
                        result += this._charMap[Zelda.MONSTER_NORMAL];
                        break;
                    case 7:
                        result += this._charMap[Zelda.MONSTER_QUICK];
                        break;
                }
            }
            result += "\n";
        }
        return result;
    }
}
Zelda.EXIT = "exit";
Zelda.AVATAR = "avatar";
Zelda.KEY = "key";
Zelda.MONSTER_QUICK = "monsterQuick";
Zelda.MONSTER_NORMAL = "monsterNormal";
Zelda.MONSTER_SLOW = "monsterSlow";
Zelda.WALL = "wall";
Zelda.EMPTY = "empty";
class Region {
    constructor(x, y, width, height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
    getVerticalSplit(minWidth) {
        if (this.width < 2 * minWidth) {
            return [this];
        }
        let split = minWidth + Math.floor(Math.random() * (this.width - 2 * minWidth));
        return [
            new Region(this.x, this.y, split, this.height),
            new Region(this.x + split, this.y, this.width - split, this.height)
        ];
    }
    getHorizontalSplit(minHeight) {
        if (this.height < 2 * minHeight) {
            return [this];
        }
        let split = minHeight + Math.floor(Math.random() * (this.height - 2 * minHeight));
        return [
            new Region(this.x, this.y, this.width, split),
            new Region(this.x, this.y + split, this.width, this.height - split)
        ];
    }
}
class BSP {
    constructor(roomWidth, roomHeight) {
        this._roomWidth = roomWidth;
        this._roomHeight = roomHeight;
    }
    shuffleArray(array) {
        for (let i = 0; i < array.length; i++) {
            let index = Math.floor(Math.random() * array.length);
            let temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
    sign(value) {
        if (value > 0) {
            return 1;
        }
        if (value < 0) {
            return -1;
        }
        return 0;
    }
    connect(map, p1, p2) {
        let dx = Math.sign(p2.x - p1.x);
        let dy = Math.sign(p2.y - p1.y);
        if (Math.random() < 0.5) {
            let x = p1.x;
            while (x != p2.x) {
                map[p1.y][x] = 0;
                x += dx;
            }
            let y = p1.y;
            while (y != p2.y) {
                map[y][p2.x] = 0;
                y += dy;
            }
        }
        else {
            let y = p1.y;
            while (y != p2.y) {
                map[y][p2.x] = 0;
                y += dy;
            }
            let x = p1.x;
            while (x != p2.x) {
                map[p1.y][x] = 0;
                x += dx;
            }
        }
    }
    generate(width, height, rooms) {
        this._width = width;
        this._height = height;
        let iterations = 0;
        let allRooms = [new Region(0, 0, this._width, this._height)];
        while (allRooms.length < rooms) {
            this.shuffleArray(allRooms);
            let before = allRooms.length;
            let currentRoom = allRooms.splice(0, 1)[0];
            let newRooms = currentRoom.getHorizontalSplit(this._roomHeight);
            if (Math.random() < 0.5 || newRooms.length == 1) {
                newRooms = currentRoom.getVerticalSplit(this._roomWidth);
            }
            allRooms = allRooms.concat(newRooms);
            let after = allRooms.length;
            if (before != after) {
                iterations = 0;
            }
            else {
                iterations += 1;
                if (iterations >= 100) {
                    break;
                }
            }
        }
        let result = [];
        for (let y = 0; y < this._height; y++) {
            result.push([]);
            for (let x = 0; x < this._width; x++) {
                result[y].push(1);
            }
        }
        for (let r of allRooms) {
            for (let y = 0; y < r.height - 1; y++) {
                for (let x = 0; x < r.width - 1; x++) {
                    let cx = Math.max(r.x + x, 1);
                    let cy = Math.max(r.y + y, 1);
                    result[cy][cx] = 0;
                }
            }
        }
        let centers = [];
        for (let i = 0; i < allRooms.length; i++) {
            for (let j = i + 1; j < allRooms.length; j++) {
                let r1 = allRooms[i];
                let r2 = allRooms[j];
                let c1 = {
                    x: r1.x + Math.floor(r1.width / 2),
                    y: r1.y + Math.floor(r1.height / 2)
                };
                let c2 = {
                    x: r2.x + Math.floor(r2.width / 2),
                    y: r2.y + Math.floor(r2.height / 2)
                };
                centers.push({
                    c1: c1, c2: c2,
                    dim: Math.sign(Math.abs(c1.x - c2.x)) + Math.sign((Math.abs(c1.y - c2.y)))
                });
            }
        }
        centers.sort((a, b) => { return a.dim - b.dim + 0.1 * (2 * Math.random() - 1); });
        for (let i = 0; i < Math.min(allRooms.length + 1, centers.length); i++) {
            let start = centers[i].c1;
            let end = centers[i].c2;
            this.connect(result, start, end);
        }
        return result;
    }
}
/// <reference path="../BSP.ts"/>
/// <reference path="../AStar.ts"/>
class CookMePasta {
    constructor(bsp, aStar, charMap) {
        this._bsp = bsp;
        this._aStar = aStar;
        this._charMap = charMap;
    }
    getDistance(p1, p2) {
        return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
    }
    getAllPossibleLocations(map) {
        let possibleLocations = [];
        for (let x = 0; x < map[0].length; x++) {
            for (let y = 0; y < map.length; y++) {
                if (map[y][x] == 0 && map[y - 1][x] == 0 && map[y + 1][x] == 0 && map[y][x - 1] == 0 && map[y][x + 1] == 0) {
                    possibleLocations.push({ x: x, y: y });
                }
            }
        }
        return possibleLocations;
    }
    getEnterHallways(map) {
        let possibleLocations = [];
        for (let x = 0; x < map[0].length; x++) {
            for (let y = 0; y < map.length; y++) {
                if (map[y][x] == 0) {
                    if (((map[y][x - 1] == 1 && map[y][x + 1] == 1) ||
                        (map[y - 1][x] == 1 && map[y + 1][x] == 1)) &&
                        this.checkNumEmpty(map, { x: x, y: y }) > 2) {
                        possibleLocations.push({ x: x, y: y });
                    }
                }
            }
        }
        return possibleLocations;
    }
    checkNumEmpty(map, p) {
        let value = 0;
        for (let y = -1; y <= 1; y++) {
            for (let x = -1; x <= 1; x++) {
                if (x == 0 && y == 0) {
                    continue;
                }
                if (map[p.y + y][p.x + x] == 0) {
                    value += 1;
                }
            }
        }
        return value;
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        let width = Math.floor((maxWidth - 14) * diff + 2 * Math.random()) + 12;
        let height = Math.floor((maxHeight - 14) * diff + 2 * Math.random()) + 12;
        let rooms = 0.5 * diff + 0.3 + 0.2 * Math.random();
        let doors = 0.7 * diff + 0.1 + 0.2 * Math.random();
        return [width, height, rooms, doors];
    }
    adjustParameters(width, height, rooms, doors) {
        let parameters = [rooms, doors];
        parameters[0] = Math.floor(rooms * 0.05 * width * height + 1);
        parameters[1] = Math.floor(doors * 0.05 * width * height);
        return [Math.max(width, 12), Math.max(height, 12)].concat(parameters);
    }
    generate(width, height, rooms, doors) {
        let map = this._bsp.generate(width, height, rooms);
        let locs = this.getAllPossibleLocations(map);
        locs.sort((a, b) => { return Math.random() - 0.5; });
        let avatar = locs.splice(0, 1)[0];
        map[avatar.y][avatar.x] = 2;
        let hallways = this.getEnterHallways(map);
        hallways.sort((a, b) => { return Math.random() - 0.5; });
        let key = false;
        for (let l of hallways) {
            if (doors == 0) {
                break;
            }
            key = true;
            map[l.y][l.x] = 3;
            doors -= 1;
        }
        if (key) {
            locs.sort((a, b) => { return this.getDistance(b, avatar) - this.getDistance(a, avatar) + 4 * (Math.random() - 0.5); });
            for (let i = 0; i < locs.length; i++) {
                if (this._aStar.pathExisits(map, avatar, locs[i])) {
                    let l = locs.splice(i, 1)[0];
                    map[l.y][l.x] = 4;
                    break;
                }
            }
        }
        let comp = 4;
        while (comp > 0) {
            locs = this.getAllPossibleLocations(map);
            locs.sort((a, b) => {
                let aReach = this._aStar.pathExisits(map, avatar, a) ? 1 : 0;
                let bReach = this._aStar.pathExisits(map, avatar, b) ? 1 : 0;
                return aReach + bReach + (Math.random() - 0.5);
            });
            let l = locs.splice(0, 1)[0];
            map[l.y][l.x] = 4 + comp;
            comp -= 1;
        }
        let result = "";
        for (let y = 0; y < map.length; y++) {
            for (let x = 0; x < map[y].length; x++) {
                switch (map[y][x]) {
                    case 0:
                        result += this._charMap[CookMePasta.EMPTY];
                        break;
                    case 1:
                        result += this._charMap[CookMePasta.WALL];
                        break;
                    case 2:
                        result += this._charMap[CookMePasta.AVATAR];
                        break;
                    case 3:
                        result += this._charMap[CookMePasta.DOOR];
                        break;
                    case 4:
                        result += this._charMap[CookMePasta.KEY];
                        break;
                    case 5:
                        result += this._charMap[CookMePasta.PASTA];
                        break;
                    case 6:
                        result += this._charMap[CookMePasta.TOMATO];
                        break;
                    case 7:
                        result += this._charMap[CookMePasta.FISH];
                        break;
                    case 8:
                        result += this._charMap[CookMePasta.BOILING];
                        break;
                }
            }
            result += "\n";
        }
        return result;
    }
}
CookMePasta.EMPTY = "empty";
CookMePasta.WALL = "wall";
CookMePasta.AVATAR = "avatar";
CookMePasta.DOOR = "door";
CookMePasta.KEY = "key";
CookMePasta.PASTA = "pasta";
CookMePasta.BOILING = "boiling";
CookMePasta.TOMATO = "tomato";
CookMePasta.FISH = "fish";
class Frogs {
    constructor(charMap) {
        this._charMap = charMap;
    }
    getDifficultyParameters(diff, maxWidth, maxHeight) {
        let width = Math.floor((maxWidth - 11) * diff + 3 * Math.random()) + 6;
        let height = Math.floor((maxHeight - 9) * diff + 3 * Math.random()) + 4;
        let streetPercentage = (1 + 0.5 * Math.random()) * diff;
        let waterPercentage = (2 + Math.random()) * diff;
        let safetyPercentage = 1 - diff;
        let maxStreetSequence = 0.3 + 0.5 * diff + 0.2 * Math.random();
        let maxWaterSequence = 0.3 + 0.5 * diff + 0.2 * Math.random();
        return [width, height, streetPercentage, waterPercentage, safetyPercentage, maxStreetSequence, maxWaterSequence];
    }
    adjustParameters(width, height, streetPercentage, waterPercentage, safetyPercentage, maxStreetSequence, maxWaterSequence) {
        let parameters = [streetPercentage, waterPercentage, safetyPercentage,
            maxStreetSequence, maxWaterSequence];
        if (streetPercentage + waterPercentage + safetyPercentage == 0) {
            parameters[0] = 1;
            parameters[1] = 1;
            parameters[2] = 1;
        }
        parameters[0] = streetPercentage / (streetPercentage + waterPercentage + safetyPercentage);
        parameters[1] = waterPercentage / (streetPercentage + waterPercentage + safetyPercentage);
        parameters[2] = safetyPercentage / (streetPercentage + waterPercentage + safetyPercentage);
        parameters[3] = Math.floor(maxStreetSequence * 0.8 * height);
        parameters[4] = Math.floor(maxStreetSequence * 0.8 * height);
        return [Math.max(width, 6) + 2, Math.max(height, 4)].concat(parameters);
    }
    getWallLine(width) {
        let result = "";
        for (let i = 0; i < width; i++) {
            result += this._charMap[Frogs.WALL];
        }
        return result;
    }
    getArray(width, base, added, num) {
        let result = [];
        for (let j = 0; j < width; j++) {
            result.push(base);
        }
        let start = Math.floor(Math.random() * (width / num)) + 1;
        let done = false;
        for (let j = 0; j < num; j++) {
            let length = Math.floor(Math.random() * 4) + 1;
            for (let l = start; l < start + length; l++) {
                if (l >= width - 1) {
                    done = true;
                    break;
                }
                result[l] = added;
            }
            start += 2 * length + 1 + Math.floor(Math.random() * (width / num));
            if (done) {
                break;
            }
        }
        return result;
    }
    generate(width, height, streetPercentage, waterPercentage, safetyPercentage, maxStreetSequence, maxWaterSequence) {
        let types = [];
        let map = [];
        for (let i = 0; i < height; i++) {
            types.push(0);
            map.push([]);
            for (let j = 0; j < width; j++) {
                map[i].push(0);
            }
        }
        for (let i = 1; i < height - 1; i++) {
            let type = 0;
            let length = 1;
            let randomValue = Math.random();
            if (randomValue < streetPercentage) {
                type = 1;
                length = Math.floor(Math.random() * maxStreetSequence) + 1;
            }
            else if (randomValue < streetPercentage + waterPercentage) {
                type = 2;
                length = Math.floor(Math.random() * maxWaterSequence) + 1;
            }
            for (let j = 0; j < length; j++) {
                if (i >= height - 1) {
                    break;
                }
                else {
                    types[i] = type;
                    if (j > 0) {
                        i += 1;
                    }
                }
            }
        }
        let goalLocation = Math.floor(Math.random() * (width - 2)) + 1;
        map[0][goalLocation] = 3;
        map[0][0] = 4;
        map[0][width - 1] = 4;
        map[0][goalLocation - 1] = 4;
        map[0][goalLocation + 1] = 4;
        let avatarLocation = Math.floor(Math.random() * (width - 2)) + 1;
        map[height - 1][avatarLocation] = 5;
        map[height - 1][0] = 4;
        map[height - 1][width - 1] = 4;
        for (let i = 1; i < height - 1; i++) {
            let num = 0;
            let state = Math.floor(Math.random() * 2);
            switch (types[i]) {
                case 0:
                    map[i][0] = 4;
                    map[i][width - 1] = 4;
                    break;
                case 1:
                    num = Math.floor(Math.random() * 0.4 * width) + 2;
                    map[i] = this.getArray(width, 1, 6 + state, num);
                    break;
                case 2:
                    for (let j = 0; j < width; j++) {
                        map[i][j] = 2;
                    }
                    num = Math.floor(Math.random() * 0.2 * width) + 1;
                    map[i] = this.getArray(width, 2, 8, num);
                    map[i][width - 2] = 8 + state;
                    map[i][width - 1] = 9;
                    break;
            }
        }
        let result = this.getWallLine(width) + "\n";
        let test = 0;
        for (let i = 0; i < map.length; i++) {
            for (let j = 0; j < map[i].length; j++) {
                switch (map[i][j]) {
                    case 0:
                        result += this._charMap[Frogs.GRASS];
                        break;
                    case 1:
                        result += this._charMap[Frogs.STREET];
                        break;
                    case 2:
                        result += this._charMap[Frogs.WATER];
                        break;
                    case 3:
                        result += this._charMap[Frogs.GOAL];
                        break;
                    case 4:
                        result += this._charMap[Frogs.WALL];
                        break;
                    case 5:
                        result += this._charMap[Frogs.AVATAR];
                        break;
                    case 6:
                        test = Math.floor(Math.random() * 2);
                        if (test == 1) {
                            result += this._charMap[Frogs.FAST_LEFT];
                        }
                        else {
                            result += this._charMap[Frogs.TRUCK_LEFT];
                        }
                        break;
                    case 7:
                        test = Math.floor(Math.random() * 2);
                        if (test == 1) {
                            result += this._charMap[Frogs.FAST_RIGHT];
                        }
                        else {
                            result += this._charMap[Frogs.TRUCK_RIGHT];
                        }
                        break;
                    case 8:
                        result += this._charMap[Frogs.LOG];
                        break;
                    case 9:
                        test = Math.floor(Math.random() * 2);
                        if (test == 1) {
                            result += this._charMap[Frogs.SPAWNER_SLOW];
                        }
                        else {
                            result += this._charMap[Frogs.SPAWNER_FAST];
                        }
                        break;
                }
            }
            result += "\n";
        }
        result += this.getWallLine(width);
        return result;
    }
}
Frogs.GRASS = "grass";
Frogs.WATER = "water";
Frogs.STREET = "street";
Frogs.TRUCK_LEFT = "truckLeft";
Frogs.TRUCK_RIGHT = "truckRight";
Frogs.FAST_LEFT = "fastLeft";
Frogs.FAST_RIGHT = "fastRight";
Frogs.WALL = "wall";
Frogs.GOAL = "goal";
Frogs.LOG = "log";
Frogs.SPAWNER_SLOW = "spawnerSlow";
Frogs.SPAWNER_FAST = "spawnerFast";
Frogs.AVATAR = "avatar";
/// <reference path="Games/Waitforbreakfast.ts"/>
/// <reference path="Games/ZenPuzzle.ts"/>
/// <reference path="Games/Boulderdash.ts"/>
/// <reference path="Games/Zelda.ts"/>
/// <reference path="Games/CookMePasta.ts"/>
/// <reference path="Games/Frogs.ts"/>
let generators = {
    "boulderdash": new Boulderdash(new CellularAutomata(5, 4, -1), new AStar([0, 2]), {
        [Boulderdash.EXIT]: 'e', [Boulderdash.AVATAR]: 'A', [Boulderdash.WALL]: 'w', [Boulderdash.DIRT]: '.',
        [Boulderdash.EMPTY]: '-', [Boulderdash.GEM]: 'x', [Boulderdash.CRAB]: 'c', [Boulderdash.BUTTERFLY]: 'b',
        [Boulderdash.BOULDER]: 'o'
    }),
    "cookmepasta": new CookMePasta(new BSP(3, 3), new AStar([0, 2]), {
        [CookMePasta.EMPTY]: '.', [CookMePasta.WALL]: 'w', [CookMePasta.PASTA]: 'p', [CookMePasta.BOILING]: 'b',
        [CookMePasta.TOMATO]: 'o', [CookMePasta.FISH]: 't', [CookMePasta.KEY]: 'k', [CookMePasta.DOOR]: 'l',
        [CookMePasta.AVATAR]: 'A'
    }),
    "waitforbreakfast": new WaitForBreakfast(new AStar([0]), {
        [WaitForBreakfast.WALL]: "w", [WaitForBreakfast.EMPTY]: ".", [WaitForBreakfast.TABLE]: "o",
        [WaitForBreakfast.TARGET_TABLE]: "t", [WaitForBreakfast.WAITER]: "k", [WaitForBreakfast.EXIT]: "e",
        [WaitForBreakfast.AVATAR]: "A", [WaitForBreakfast.LEFT]: "r", [WaitForBreakfast.RIGHT]: "l",
        [WaitForBreakfast.UP]: "f", [WaitForBreakfast.DOWN]: "b", [WaitForBreakfast.TARGET_LEFT]: "3",
        [WaitForBreakfast.TARGET_RIGHT]: "2", [WaitForBreakfast.TARGET_UP]: "0", [WaitForBreakfast.TARGET_DOWN]: "1"
    }),
    "zelda": new Zelda(new Maze(), {
        [Zelda.EMPTY]: '.', [Zelda.WALL]: 'w', [Zelda.KEY]: '+', [Zelda.AVATAR]: 'A', [Zelda.EXIT]: 'g',
        [Zelda.MONSTER_SLOW]: '1', [Zelda.MONSTER_NORMAL]: '2', [Zelda.MONSTER_QUICK]: '3'
    }),
    "zenpuzzle": new ZenPuzzle(require('hilbert').Hilbert2d, {
        [ZenPuzzle.AVATAR]: "A", [ZenPuzzle.EMPTY]: ".",
        [ZenPuzzle.TILE]: "g", [ZenPuzzle.ROCK]: "r"
    }),
    "frogs": new Frogs({
        [Frogs.GRASS]: "+", [Frogs.STREET]: ".", [Frogs.WATER]: "0", [Frogs.LOG]: "=", [Frogs.AVATAR]: "A",
        [Frogs.TRUCK_LEFT]: "_", [Frogs.FAST_LEFT]: "l", [Frogs.TRUCK_RIGHT]: "-", [Frogs.FAST_RIGHT]: "x",
        [Frogs.WALL]: "w", [Frogs.SPAWNER_FAST]: "1", [Frogs.SPAWNER_SLOW]: "3", [Frogs.GOAL]: "g"
    })
};
let parameterSize = {
    "boulderdash": 5,
    "cookmepasta": 2,
    "waitforbreakfast": 1,
    "zelda": 3,
    "zenpuzzle": 2,
    "frogs": 5
};
let paddingChars = {
    "boulderdash": 'w',
    "cookmepasta": 'w',
    "waitforbreakfast": 'w',
    "zelda": 'w',
    "zenpuzzle": '.',
    "frogs": 'w'
};
function padLevels(level, maxWidth, maxHeight, character) {
    if (maxWidth <= 0 || maxHeight <= 0) {
        return level;
    }
    let levelLines = level.split("\n");
    for (let i = 0; i < levelLines.length; i++) {
        if (levelLines[i].trim().length == 0) {
            levelLines.splice(i, 1);
            i--;
        }
    }
    let padHorizontal = maxWidth - levelLines[0].length;
    let padVertical = maxHeight - levelLines.length;
    let padLeft = Math.floor(padHorizontal / 2);
    let padRight = padHorizontal - padLeft;
    let padTop = Math.floor(padVertical / 2);
    let padBot = padVertical - padTop;
    let result = "";
    for (let i = 0; i < padTop; i++) {
        for (let j = 0; j < maxWidth; j++) {
            result += character;
        }
        result += "\n";
    }
    for (let i = 0; i < levelLines.length; i++) {
        if (levelLines[i].trim().length == 0) {
            continue;
        }
        for (let j = 0; j < padLeft; j++) {
            result += character;
        }
        result += levelLines[i];
        for (let j = 0; j < padRight; j++) {
            result += character;
        }
        result += "\n";
    }
    for (let i = 0; i < padBot; i++) {
        for (let j = 0; j < maxWidth; j++) {
            result += character;
        }
        result += "\n";
    }
    return result;
}
let game = process.argv[2];
let levelFile = process.argv[3];
let maxWidth = 0;
let maxHeight = 0;
let parameters = [];
if (process.argv[4] != "difficulty") {
    let width = parseInt(process.argv[4]);
    let height = parseInt(process.argv[5]);
    for (let i = 0; i < parameterSize[game]; i++) {
        if (i + 6 < process.argv.length) {
            parameters.push(parseFloat(process.argv[i + 6]));
        }
        else {
            parameters.push(Math.random());
        }
    }
    parameters = [width, height].concat(parameters);
}
else {
    maxWidth = parseInt(process.argv[6]);
    maxHeight = parseInt(process.argv[7]);
    parameters = generators[game].getDifficultyParameters(parseFloat(process.argv[5]), maxWidth, maxHeight);
    // console.log(parameters);
}
parameters = generators[game].adjustParameters.apply(generators[game], parameters);
let level = generators[game].generate.apply(generators[game], parameters);
level = padLevels(level, maxWidth, maxHeight, paddingChars[game]);
let fs = require('fs');
fs.writeFileSync(levelFile, level);
//# sourceMappingURL=app_v2.js.map
