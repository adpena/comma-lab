// SPDX-License-Identifier: MIT
// Runtime-only private-ANE surface inventory.  This probe does not forge an
// entitlement, instantiate unknown private objects, or claim execution.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static NSString *JSONEscape(NSString *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString *array = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

static NSArray<NSString *> *MethodsForClass(Class cls, BOOL classMethods) {
    Class target = classMethods ? object_getClass(cls) : cls;
    unsigned int count = 0;
    Method *methods = class_copyMethodList(target, &count);
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        [names addObject:NSStringFromSelector(method_getName(methods[i]))];
    }
    free(methods);
    [names sortUsingSelector:@selector(compare:)];
    return names;
}

int main(void) {
    @autoreleasepool {
        NSArray<NSString *> *paths = @[
            @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            @"/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler"
        ];
        NSMutableArray *frameworks = [NSMutableArray array];
        for (NSString *path in paths) {
            dlerror();
            void *handle = dlopen(path.UTF8String, RTLD_NOW | RTLD_LOCAL);
            const char *error = dlerror();
            [frameworks addObject:@{
                @"path": path,
                @"loaded": @(handle != NULL),
                @"dlerror": error ? [NSString stringWithUTF8String:error] : @""
            }];
        }

        int count = objc_getClassList(NULL, 0);
        Class *classes = count > 0 ? calloc((size_t)count, sizeof(Class)) : NULL;
        count = classes ? objc_getClassList(classes, count) : 0;
        NSMutableArray *inventory = [NSMutableArray array];
        NSMutableArray *execution = [NSMutableArray array];
        NSMutableArray *backward = [NSMutableArray array];
        NSArray<NSString *> *executionTokens = @[@"evaluate", @"execute", @"enqueue", @"infer", @"prediction"];
        NSArray<NSString *> *backwardTokens = @[
            @"backward", @"gradient", @"vjp", @"adjoint", @"autodiff", @"train",
            @"derivative", @"differentiate", @"costate", @"reverse", @"backprop"
        ];
        for (int i = 0; i < count; ++i) {
            NSString *name = NSStringFromClass(classes[i]);
            if (!([name hasPrefix:@"_ANE"] || [name hasPrefix:@"ANE"])) continue;
            NSArray *instanceMethods = MethodsForClass(classes[i], NO);
            NSArray *classMethods = MethodsForClass(classes[i], YES);
            [inventory addObject:@{
                @"class": name,
                @"instance_methods": instanceMethods,
                @"class_methods": classMethods
            }];
            for (NSString *method in [instanceMethods arrayByAddingObjectsFromArray:classMethods]) {
                NSString *lower = method.lowercaseString;
                for (NSString *token in executionTokens) {
                    if ([lower rangeOfString:token].location != NSNotFound) {
                        [execution addObject:@{@"class": name, @"selector": method, @"token": token}];
                        break;
                    }
                }
                for (NSString *token in backwardTokens) {
                    if ([lower rangeOfString:token].location != NSNotFound) {
                        [backward addObject:@{@"class": name, @"selector": method, @"token": token}];
                        break;
                    }
                }
            }
        }
        free(classes);
        [inventory sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
            return [a[@"class"] compare:b[@"class"]];
        }];
        NSDictionary *output = @{
            @"schema": @"ane_private_runtime_inventory.v1",
            @"frameworks": frameworks,
            @"ane_classes": inventory,
            @"execution_selector_candidates": execution,
            @"backward_selector_candidates": backward,
            @"direct_forward_executed": @NO,
            @"backward_vjp_executed": @NO,
            @"safety_scope": @"introspection only; no entitlement forgery or unknown private-object invocation"
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:output options:NSJSONWritingPrettyPrinted error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
    }
    return 0;
}
