import argparse
import os
import acl
import numpy as np

HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
EXPECTED_NUM_CLASSES = 20
EXPECTED_BBOX_CH = 4
EXPECTED_TOTAL_CH = EXPECTED_BBOX_CH + EXPECTED_NUM_CLASSES
EXPECTED_ANCHORS = 8400


def check_ret(name, ret):
    if ret != 0:
        raise RuntimeError(f"{name} failed ret={ret}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_shape(dims_dict):
    dims = dims_dict.get("dims", [])
    return [int(x) for x in dims]


def get_output_dims(model_desc, index):
    if hasattr(acl.mdl, "get_output_dims"):
        dims, ret = acl.mdl.get_output_dims(model_desc, index)
        check_ret("acl.mdl.get_output_dims", ret)
        return safe_shape(dims)
    return []


def print_array_stats(name, arr, head=20):
    flat = arr.reshape(-1)
    print(f"\n[{name}]")
    print(f"  shape={list(arr.shape)} dtype={arr.dtype} numel={flat.size}")
    if flat.size == 0:
        print("  empty tensor")
        return
    print(
        "  min={:.6f} max={:.6f} mean={:.6f} std={:.6f}".format(
            float(flat.min()), float(flat.max()), float(flat.mean()), float(flat.std())
        )
    )
    print(f"  first_{min(head, flat.size)}={flat[:head]}")


def analyze_view_chw(arr3):
    # arr3: [B, C, N]
    bbox = arr3[0, :4, :]
    cls = arr3[0, 4:, :]
    print("  layout_guess=CHW [B,C,N]")
    print(
        "  bbox_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(bbox.min()), float(bbox.max()), float(bbox.mean())
        )
    )
    print(
        "  cls_raw_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(cls.min()), float(cls.max()), float(cls.mean())
        )
    )
    print(
        "  cls_sigmoid_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(sigmoid(cls).min()), float(sigmoid(cls).max()), float(sigmoid(cls).mean())
        )
    )
    for i in range(min(5, bbox.shape[1])):
        raw_bbox = bbox[:, i]
        cls_raw = cls[:, i]
        top_idx = np.argsort(cls_raw)[-3:][::-1]
        top_desc = [
            f"cls={int(k)} raw={float(cls_raw[k]):.6f} sig={float(sigmoid(cls_raw[k])):.6f}"
            for k in top_idx
        ]
        print(
            f"  anchor[{i}] bbox(raw)={raw_bbox.tolist()} top3={top_desc}"
        )


def analyze_view_hwc(arr3):
    # arr3: [B, N, C]
    bbox = arr3[0, :, :4]
    cls = arr3[0, :, 4:]
    print("  layout_guess=HWC [B,N,C]")
    print(
        "  bbox_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(bbox.min()), float(bbox.max()), float(bbox.mean())
        )
    )
    print(
        "  cls_raw_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(cls.min()), float(cls.max()), float(cls.mean())
        )
    )
    print(
        "  cls_sigmoid_range(min/max/mean)=({:.6f}, {:.6f}, {:.6f})".format(
            float(sigmoid(cls).min()), float(sigmoid(cls).max()), float(sigmoid(cls).mean())
        )
    )
    for i in range(min(5, bbox.shape[0])):
        raw_bbox = bbox[i, :]
        cls_raw = cls[i, :]
        top_idx = np.argsort(cls_raw)[-3:][::-1]
        top_desc = [
            f"cls={int(k)} raw={float(cls_raw[k]):.6f} sig={float(sigmoid(cls_raw[k])):.6f}"
            for k in top_idx
        ]
        print(
            f"  anchor[{i}] bbox(raw)={raw_bbox.tolist()} top3={top_desc}"
        )


def analyze_yoloworld_like(output, tensor_name):
    flat = output.reshape(-1)
    total = flat.size
    expected_total = EXPECTED_TOTAL_CH * EXPECTED_ANCHORS

    print(f"\n[{tensor_name}] layout/bbox/class analysis")

    if total != expected_total:
        print(
            f"  skip special analysis: numel={total}, expected={expected_total} "
            f"for {EXPECTED_TOTAL_CH}x{EXPECTED_ANCHORS}"
        )
        return

    views = []
    if output.ndim == 3 and tuple(output.shape[1:]) == (EXPECTED_TOTAL_CH, EXPECTED_ANCHORS):
        views.append(("shape_native", "CHW", output))
    if output.ndim == 3 and tuple(output.shape[1:]) == (EXPECTED_ANCHORS, EXPECTED_TOTAL_CH):
        views.append(("shape_native", "HWC", output))

    chw_guess = flat.reshape(1, EXPECTED_TOTAL_CH, EXPECTED_ANCHORS)
    hwc_guess = flat.reshape(1, EXPECTED_ANCHORS, EXPECTED_TOTAL_CH)
    views.append(("reinterpret", "CHW", chw_guess))
    views.append(("reinterpret", "HWC", hwc_guess))

    printed = set()
    for source, layout, arr3 in views:
        key = (source, layout, tuple(arr3.shape))
        if key in printed:
            continue
        printed.add(key)
        print(f"  source={source} shape={list(arr3.shape)}")
        if layout == "CHW":
            analyze_view_chw(arr3)
        else:
            analyze_view_hwc(arr3)


def main():
    parser = argparse.ArgumentParser(description="Inspect Ascend OM outputs for YOLOWorld postprocess design.")
    parser.add_argument("--model", required=True, help="Path to .om model")
    parser.add_argument("--device", type=int, default=0, help="Ascend device id")
    parser.add_argument("--fill", type=float, default=0.5, help="Constant input fill value")
    parser.add_argument("--save-dir", default="demo/out/om_inspect", help="Directory to save .npy outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ret = acl.init()
    check_ret("acl.init", ret)

    ret = acl.rt.set_device(args.device)
    check_ret("acl.rt.set_device", ret)

    context, ret = acl.rt.create_context(args.device)
    check_ret("acl.rt.create_context", ret)

    model_id = None
    model_desc = None
    input_dataset = None
    output_dataset = None
    input_buffers = []
    output_buffers = []
    host_ptrs = []

    try:
        model_id, ret = acl.mdl.load_from_file(args.model)
        check_ret("acl.mdl.load_from_file", ret)

        model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(model_desc, model_id)
        check_ret("acl.mdl.get_desc", ret)

        input_num = acl.mdl.get_num_inputs(model_desc)
        output_num = acl.mdl.get_num_outputs(model_desc)
        print(f"Input num: {input_num}")
        print(f"Output num: {output_num}")

        input_dataset = acl.mdl.create_dataset()
        for i in range(input_num):
            dims, ret = acl.mdl.get_input_dims(model_desc, i)
            check_ret("acl.mdl.get_input_dims", ret)
            shape = safe_shape(dims)
            print(f"Input[{i}] shape={shape}")

            data = np.full(shape, args.fill, dtype=np.float32)
            size = data.nbytes
            device_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)
            ret = acl.rt.memcpy(
                device_ptr,
                size,
                acl.util.numpy_to_ptr(data),
                size,
                HOST_TO_DEVICE,
            )
            check_ret("acl.rt.memcpy(HOST_TO_DEVICE)", ret)
            buffer = acl.create_data_buffer(device_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(input_dataset, buffer)
            check_ret("acl.mdl.add_dataset_buffer(input)", ret)
            input_buffers.append((device_ptr, buffer))

        output_dataset = acl.mdl.create_dataset()
        output_shapes = []
        for i in range(output_num):
            shape = get_output_dims(model_desc, i)
            size = acl.mdl.get_output_size_by_index(model_desc, i)
            print(f"Output[{i}] shape_hint={shape} bytes={size}")
            device_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)
            buffer = acl.create_data_buffer(device_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(output_dataset, buffer)
            check_ret("acl.mdl.add_dataset_buffer(output)", ret)
            output_buffers.append((device_ptr, size, buffer))
            output_shapes.append(shape)

        ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
        check_ret("acl.mdl.execute", ret)

        print("\n===== Inference outputs =====")
        for i, (device_ptr, size, _) in enumerate(output_buffers):
            host_ptr, ret = acl.rt.malloc_host(size)
            check_ret("acl.rt.malloc_host", ret)
            host_ptrs.append(host_ptr)

            ret = acl.rt.memcpy(
                host_ptr,
                size,
                device_ptr,
                size,
                DEVICE_TO_HOST,
            )
            check_ret("acl.rt.memcpy(DEVICE_TO_HOST)", ret)

            data_bytes = acl.util.ptr_to_bytes(host_ptr, size)
            output = np.frombuffer(data_bytes, dtype=np.float32).copy()

            shape = output_shapes[i]
            if shape and np.prod(shape) == output.size:
                output = output.reshape(shape)
                shape_status = "reshaped_by_model_desc"
            else:
                shape_status = "flat_fallback"

            print(f"\nOutput[{i}] read_status={shape_status}")
            print_array_stats(f"output_{i}", output)
            analyze_yoloworld_like(output, f"output_{i}")

            save_path = os.path.join(args.save_dir, f"output_{i}.npy")
            np.save(save_path, output)
            print(f"  saved_npy={save_path}")

    finally:
        if input_dataset is not None:
            acl.mdl.destroy_dataset(input_dataset)
        if output_dataset is not None:
            acl.mdl.destroy_dataset(output_dataset)

        for device_ptr, buffer in input_buffers:
            try:
                acl.destroy_data_buffer(buffer)
            except Exception:
                pass
            try:
                acl.rt.free(device_ptr)
            except Exception:
                pass

        for device_ptr, _, buffer in output_buffers:
            try:
                acl.destroy_data_buffer(buffer)
            except Exception:
                pass
            try:
                acl.rt.free(device_ptr)
            except Exception:
                pass

        for host_ptr in host_ptrs:
            try:
                acl.rt.free_host(host_ptr)
            except Exception:
                pass

        if model_desc is not None:
            try:
                acl.mdl.destroy_desc(model_desc)
            except Exception:
                pass
        if model_id is not None:
            try:
                acl.mdl.unload(model_id)
            except Exception:
                pass

        try:
            acl.rt.destroy_context(context)
        except Exception:
            pass
        try:
            acl.rt.reset_device(args.device)
        except Exception:
            pass
        try:
            acl.finalize()
        except Exception:
            pass


if __name__ == "__main__":
    main()
